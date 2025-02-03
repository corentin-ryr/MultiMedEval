from collections import defaultdict


# def recursive_modifier(annotations, ind, d):
#     modifiers = []
#     if ind in d:
#         for modifier_index in d[ind]:
#             modifiers += recursive_modifier(annotations, modifier_index, d)
#     modifiers.append((annotations[ind]["tokens"], annotations[ind]["label"], annotations[ind]["start_ix"],
#                       annotations[ind]["end_ix"]))
#     return modifiers


def recursive_modifier(annotations, ind, d, visited=None):
    if visited is None:
        visited = set()
    # If we have already visited this entity, we have a cycle
    if ind in visited:
        # Return empty list or break out to avoid infinite recursion
        return []
    visited.add(ind)

    modifiers = []
    # If this entity modifies others, recursively gather their modifiers
    if ind in d:
        for modifier_index in d[ind]:
            modifiers += recursive_modifier(annotations, modifier_index, d, visited)
    # Add this entity's annotation
    modifiers.append((annotations[ind]["tokens"], annotations[ind]["label"], annotations[ind]["start_ix"],
                      annotations[ind]["end_ix"]))
    return modifiers


def sort_words_by_index(word_list, index_list):
    sorted_words = [word_list[index_list.index(i)] for i in sorted(index_list)]
    return sorted_words


def filter_loop(d):
    to_keep = defaultdict(list)
    for k, v in d.items():
        opposite = (v[0], [k])
        if opposite not in to_keep.items():
            to_keep[k] = v
    return to_keep


def get_radgraph_processed_annotations(radgraph_annotations):
    annotations: dict = radgraph_annotations["0"]["entities"]
    radgraph_text = radgraph_annotations["0"]["text"]

    obs_modified_by_obs = defaultdict(list)
    obs_located_anat = defaultdict(list)
    obs_suggest_obs = defaultdict(list)
    anat_modify_anat = defaultdict(list)
    all_observations = []

    tags_name = {
        "Observation::uncertain": 'uncertain',
        "Observation::definitely present": 'definitely present',
        "Observation::definitely absent": 'definitely absent',
    }

    # First loop over all entities
    for index, v in annotations.items():
        label = v["label"]
        tokens = v["tokens"]
        relations = v["relations"]

        # We consider a "main observation" an Observation entity that does not modify
        if 'Observation' in label and not ('modify' in [relation[0] for relation in relations]):
            all_observations.append((index, tokens, label))

        # For the current entity, fill the following dict
        for relation in relations:
            if 'modify' in relation and 'Observation' in label:
                target = relation[1]
                obs_modified_by_obs[target].append(index)

            if 'modify' in relation and 'Anatomy' in label:
                target = relation[1]
                anat_modify_anat[target].append(index)

            if 'located_at' in relation and 'Observation' in label:
                target = relation[1]
                obs_located_anat[index].append(target)

            if 'suggestive_of' in relation and 'Observation' in label:
                target = relation[1]
                obs_suggest_obs[index].append(target)

    # filtering loop
    obs_modified_by_obs = filter_loop(obs_modified_by_obs)
    anat_modify_anat = filter_loop(anat_modify_anat)

    processed_observations = []
    # For each main observation
    for observation in all_observations:
        record = {
            "observation": None,
            "observation_start_ix": [],
            "located_at": [],
            "located_at_start_ix": [],
            "tags": [],
            "suggestive_of": None
        }
        observation_index, tokens, label = observation

        # Recursively get full observation name with modifiers (such as increased)
        # We also return the labels, and start_ix (index in the sentence) of retrieved entities
        modifiers = recursive_modifier(annotations, observation_index, obs_modified_by_obs)

        modifiers_labels = [m[1] for m in modifiers]
        modifiers_start_ix = [int(m[2]) for m in modifiers]
        modifiers_end_ix = [int(m[3]) for m in modifiers]
        modifiers_tokens = [m[0] for m in modifiers]

        # We rearrange the words according to start_ix (order they appear in the sentence)
        # for example wall abdominal -> adbominal wall
        modifiers_tokens = sort_words_by_index(modifiers_tokens, modifiers_start_ix)
        modifiers_start_ix = sorted(modifiers_start_ix)
        modifiers_end_ix = sorted(modifiers_end_ix)

        # Sometimes, because of recursivity, we fetch twice the same modifiers (two modifiers have the same modifiers)
        # Need to filter
        modifiers_tokens = [x for i, x in enumerate(modifiers_tokens) if x not in modifiers_tokens[:i]]
        modifiers_start_ix = [x for i, x in enumerate(modifiers_start_ix) if
                              x not in modifiers_start_ix[:i]]
        modifiers_end_ix = [x for i, x in enumerate(modifiers_end_ix) if
                            x not in modifiers_end_ix[:i]]

        record["observation"] = " ".join(modifiers_tokens).lower().strip("\n .\"'")
        record["observation_start_ix"] = modifiers_start_ix
        record["observation_end_ix"] = modifiers_end_ix

        # We prepend "uncertain" or "no" to the full observation if the obs or modifiers contain the corresponding label
        # This is working well.
        if "Observation::definitely absent" in modifiers_labels:
            # record["observation"] = "no " + record["observation"]
            record["observation"] = record["observation"]
            tag = tags_name["Observation::definitely absent"]
        elif "Observation::uncertain" in modifiers_labels:
            # record["observation"] = "possible " + record["observation"]
            record["observation"] = record["observation"]
            tag = tags_name["Observation::uncertain"]
        else:
            tag = tags_name["Observation::definitely present"]

        # Tag
        record["tags"] = [tag]

        # We do the exact same for the anatomies of main observations (in obs_located_anat)
        if observation_index in obs_located_anat:

            located_at = []
            located_at_start_ix = []
            located_at_end_ix = []
            anats_index = obs_located_anat[observation_index]

            # recursively retrieve modifiers
            for anat_index in anats_index:
                modifiers = recursive_modifier(annotations, anat_index, anat_modify_anat)
                modifiers_start_ix = [int(m[2]) for m in modifiers]
                modifiers_end_ix = [int(m[3]) for m in modifiers]
                modifiers_tokens = [m[0] for m in modifiers]

                # Sorting
                modifiers_tokens = " ".join(sort_words_by_index(modifiers_tokens, modifiers_start_ix))
                modifiers_start_ix = sorted(modifiers_start_ix)
                modifiers_end_ix = sorted(modifiers_end_ix)

                # if len(modifiers_tokens.split(" ")) > 1:
                #     modifiers_tokens = correct_modifier_order(modifiers_tokens)

                located_at.append(modifiers_tokens.lower().strip("\n .\"'"))
                located_at_start_ix.append(modifiers_start_ix)
                located_at_end_ix.append(modifiers_end_ix)

            # List of "located at" anat for current observation, filtering duplicates
            # record["located_at"] = [x for i, x in enumerate(located_at) if x not in located_at[:i]]
            # record["located_at_start_ix"] = [x for i, x in enumerate(located_at_start_ix) if
            #                                  x not in located_at_start_ix[:i]]

            record["located_at"] = located_at
            record["located_at_start_ix"] = located_at_start_ix

        # Suggestive of
        if observation_index in obs_suggest_obs:
            targets = obs_suggest_obs[observation_index]
            suggestive_of_records = []
            for target in targets:
                suggestive_of_records.append(tokens + " suggestive of " + annotations[target]["tokens"])

            record["suggestive_of"] = suggestive_of_records

        processed_observations.append(record)

    start_ix_to_label = {v["start_ix"]: v["label"] for v in radgraph_annotations["0"]["entities"].values()}

    return {"processed_annotations": processed_observations,
            "radgraph_annotations": radgraph_annotations,
            "start_ix_to_label": start_ix_to_label,
            "radgraph_text": radgraph_text}

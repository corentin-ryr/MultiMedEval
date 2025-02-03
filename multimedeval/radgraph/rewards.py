def exact_entity_token_if_all_match_reward(
        hypothesis_annotation_list, reference_annotation_list
):
    candidates = []
    for annotation_list in [hypothesis_annotation_list, reference_annotation_list]:
        candidate = []
        for entity in annotation_list["entities"].values():
            if not entity["relations"]:
                candidate.append((entity["tokens"], entity["label"]))
            if entity["relations"]:
                candidate.extend([(entity["tokens"].lower(),
                                   entity["label"],
                                   r[0],
                                   annotation_list["entities"][r[1]]["tokens"].lower())
                                  for r in entity["relations"]]
                                 )

        candidate = set(candidate)
        candidates.append(candidate)

    hypothesis_relation_token_list, reference_relation_token_list = candidates
    precision = (
        sum(
            [
                1
                for x in hypothesis_relation_token_list
                if (x in reference_relation_token_list)
            ]
        )
        / len(hypothesis_relation_token_list)
        if len(hypothesis_relation_token_list) > 0
        else 0.0
    )
    recall = (
        sum(
            [
                1
                for x in reference_relation_token_list
                if (x in hypothesis_relation_token_list)
            ]
        )
        / len(reference_relation_token_list)
        if len(reference_relation_token_list) > 0
        else 0.0
    )
    f1_score = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return f1_score


def exact_entity_token_if_rel_exists_reward(
        hypothesis_annotation_list, reference_annotation_list
):
    candidates = []
    for annotation_list in [hypothesis_annotation_list, reference_annotation_list]:
        candidate = []
        for entity in annotation_list["entities"].values():
            if not entity["relations"]:
                candidate.append((entity["tokens"], entity["label"]))
            if entity["relations"]:
                candidate.append((entity["tokens"], entity["label"], True))

        candidate = set(candidate)
        candidates.append(candidate)

    hypothesis_relation_token_list, reference_relation_token_list = candidates

    precision = (
        sum(
            [
                1
                for x in hypothesis_relation_token_list
                if (x in reference_relation_token_list)
            ]
        )
        / len(hypothesis_relation_token_list)
        if len(hypothesis_relation_token_list) > 0
        else 0.0
    )
    recall = (
        sum(
            [
                1
                for x in reference_relation_token_list
                if (x in hypothesis_relation_token_list)
            ]
        )
        / len(reference_relation_token_list)
        if len(reference_relation_token_list) > 0
        else 0.0
    )
    f1_score = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return f1_score


def exact_entity_token_match_reward(
        hypothesis_annotation_list, reference_annotation_list
):
    candidates = []
    for annotation_list in [hypothesis_annotation_list, reference_annotation_list]:
        candidate = []
        for entity in annotation_list["entities"].values():
            candidate.append((entity["tokens"], entity["label"]))

        candidate = set(candidate)
        candidates.append(candidate)

    hypothesis_relation_token_list, reference_relation_token_list = candidates

    precision = (
        sum(
            [
                1
                for x in hypothesis_relation_token_list
                if (x in reference_relation_token_list)
            ]
        )
        / len(hypothesis_relation_token_list)
        if len(hypothesis_relation_token_list) > 0
        else 0.0
    )
    recall = (
        sum(
            [
                1
                for x in reference_relation_token_list
                if (x in hypothesis_relation_token_list)
            ]
        )
        / len(reference_relation_token_list)
        if len(reference_relation_token_list) > 0
        else 0.0
    )

    f1_score = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return f1_score


def compute_reward(
        hypothesis_annotation_list,
        reference_annotation_list,
        reward_level,
):
    assert reward_level in ["simple", "partial", "complete", "all"]
    if (
            len(hypothesis_annotation_list["entities"].keys()) == 0
            or len(reference_annotation_list["entities"].keys()) == 0
    ):
        return (0., 0., 0.) if reward_level == "all" else 0.
    simple = exact_entity_token_match_reward(hypothesis_annotation_list, reference_annotation_list)
    partial = exact_entity_token_if_rel_exists_reward(hypothesis_annotation_list, reference_annotation_list)
    complete = exact_entity_token_if_all_match_reward(hypothesis_annotation_list, reference_annotation_list)
    all = (simple, partial, complete)
    return eval(reward_level)

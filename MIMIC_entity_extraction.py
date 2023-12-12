import os
import sys
import json
import gdown
import pandas as pd
import csv
import datasets
from tqdm import tqdm
import math
from multimedbench.utils import batchSampler, remove_punctuation

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches



class MIMIC_entity_extraction:
    def __init__(self):
        self._prepare_radgraph()

        # Get the dataset ====================================================================
        self.path = json.load(open("MedMD_config.json", "r"))["MIMIC-CXR"]["path"]

        # Get the split.csv file in the image directory
        split = pd.read_csv(os.path.join(self.path, "mimic-cxr-2.0.0-split.csv"))
        split = split[split.split == "test"]

        findings = {}
        with open(os.path.join(self.path, "mimic_findings.csv"), "r") as f:
            spamreader = csv.reader(f, delimiter=",")
            for line in spamreader:
                findings[line[0]] = line[1]

        # For each sample in the dataset, open its report and check if it contains the keyword "findings"
        self.dataset = []
        for _, row in split.iterrows():
            reportFindings = findings["s" + str(row["study_id"]) + ".txt"]
            if reportFindings == "NO FINDINGS" or reportFindings == "":
                continue

            self.dataset.append(
                [str(row["subject_id"]), str(row["study_id"]), str(row["dicom_id"]), str(reportFindings)]
            )

        self.dataset = datasets.Dataset.from_pandas(
            pd.DataFrame(columns=["subject_id", "study_id", "dicom_id", "findings"], data=self.dataset)
        )

    def _prepare_radgraph(self):
        # Open the MedMD_config json file and get the download location for radgraph
        with open("MedMD_config.json", "r") as f:
            output = json.load(f)["RadGraph"]["dlLocation"]

        if not os.path.exists(os.path.join(output, "scorers")):
            gdown.download("https://drive.google.com/uc?id=1koePS_rgP5_zNUeqnQgdQ89nQEolTEbR", output, quiet=False)

            # Unzip the archive and delete the archive
            import zipfile

            with zipfile.ZipFile(os.path.join(output, "scorers.zip"), "r") as zip_ref:
                zip_ref.extractall(os.path.join(output, "scorers"))
            os.remove(os.path.join(output, "scorers.zip"))
        else:
            print("RadGraph already downloaded")

        # Add the RadGraph to the path
        sys.path.append(output)

        try:
            from scorers.RadGraph.RadGraph import (
                RadGraph,
            )  # It is normal that the import is not found by the IDE because it will be downloaded and installed at runtime
        except Exception as e:
            print("There was an error during the download and install of RadGraph")
            raise e

        self.radgraph = RadGraph(reward_level="partial")

    def create_reports_graph(self, ref, hyp):
        # Compute the F1-radgraph score
        (_, _, hyp_annotation_lists, ref_annotation_lists) = self.radgraph(refs=[ref], hyps=[hyp])

        return hyp_annotation_lists[0]["entities"]

    def run(self):

        entitiesAndSentences = []

        for batch in tqdm(
            batchSampler(self.dataset, 4),
            total=math.ceil(len(self.dataset) / 4),
            desc="Generating reports",
        ):
            refReports = [sample["findings"] for sample in batch]

            for report in refReports:
                entities = self.create_reports_graph(report, report)


                entitiesAndSentences.append([report, entities])

                # self.dependency_parser(report)

        # Write the output to a file
        with open("entitiesAndSentences.json", "w") as f:
            json.dump(entitiesAndSentences, f)


    def dependency_parser(self, report):
        import spacy
        from spacy import displacy

        # Load the language model
        nlp = spacy.load("en_core_web_sm")

        # nlp function returns an object with individual token information,
        # linguistic features and relationships
        doc = nlp(report)

        print("{:<15} | {:<8} | {:<15} | {:<20}".format("Token", "Relation", "Head", "Children"))
        print("-" * 70)

        for token in doc:
            # Print the token, dependency nature, head and all dependents of the token
            print(
                "{:<15} | {:<8} | {:<15} | {:<20}".format(
                    str(token.text), str(token.dep_), str(token.head.text), str([child for child in token.children])
                )
            )

        # Use displayCy to visualize the dependency
        svg = displacy.render(doc, style="dep", options={"distance": 120})

        # Write the output to a file
        with open("dependency.svg", "w") as f:
            f.write(svg)

def plot_sentence(sentence, relations):
    fig, ax = plt.subplots()

    # Set the size of the plot
    fig.set_size_inches(15, 4)

    sentence = [remove_punctuation(word) for word in sentence.split()]

    print(sentence)

    # Plot the sentence
    for i, word in enumerate(sentence):
        ax.text(i * 0.01, 0.5, word, va='center', ha='center', fontsize=9, color='black')
    

    # Highlight the specified words
    for relation in relations:
        # Find the position of each word in the sentence
        word1 = sentence.index(relation[0])
        word2 = sentence.index(relation[1])

        a = patches.FancyArrowPatch((word1 * 0.01, 0.45), (word2 * 0.01, 0.45),
                             connectionstyle="arc3,rad=.5")
        plt.gca().add_patch(a)

    # Remove x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()


entitiesAndSentences = json.load(open("entitiesAndSentences.json", "r"))

# entities = entitiesAndSentences[0][1]

locatedAtRelations = []
locatedAtAndObservationRelations = []
modifyAndObservationRelations = []
for entities in entitiesAndSentences:
    entities = entities[1]
    relations = []
    for entity in entities:
        currentEntity = entities[entity]

        if currentEntity["relations"] == []:
            continue

        for relation in currentEntity["relations"]:
            otherEntity = entities[relation[1]]
            currentRelation = (currentEntity["tokens"], otherEntity["tokens"])
            relations.append(currentRelation)

            # Keep the entities with a located at relation
            if relation[0] == "located_at":
                locatedAtRelations.append(currentRelation)

                if currentEntity["label"] == "OBS-DP":
                    locatedAtAndObservationRelations.append(currentRelation)
            
            if relation[0] == "modify" and currentEntity["label"] == "OBS-DP":
                modifyAndObservationRelations.append(currentRelation)

# print(locatedAtRelations)
json.dump(modifyAndObservationRelations, open("locatedAtRelations.json", "w"))


# plot_sentence(entitiesAndSentences[0][0], relations)

# extractor = MIMIC_entity_extraction()
# extractor.run()





from multimedbench.utils import Benchmark
import pandas as pd
from pathlib import Path
import math


class BenchmarkVisualizer:
    def __init__(self, datasets: list[Benchmark]) -> None:
        self.datasets = datasets

        self.folderName = "dataset_stats"

        # Create the folder if it doesn't exist
        Path(self.folderName).mkdir(parents=True, exist_ok=True)

    def sunburstModalities(self):
        print("======================= Creating sunburst modalities =======================")
        import plotly.express as px

        title = "Modality"
        totalSamples = 0

        # Create a dataframe with column "modality", "task", "dataset" and "size"
        data = []
        for dataset in self.datasets:
            data.append(
                {
                    "title": title,
                    "modality": dataset.modality,
                    "task": dataset.task,
                    "dataset": dataset.taskName,
                    "size": len(dataset),
                }
            )
            totalSamples += len(dataset)
        df = pd.DataFrame(columns=["title", "modality", "task", "dataset", "size"], data=data)

        fig = px.sunburst(df, path=["title", "modality", "task", "dataset"], values="size")
        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

        # fig.update_layout(title_text=f"Modality (Total samples: {totalSamples})")
        fig.write_image(Path(self.folderName, "modalities.png"), scale=1.0, width=750, height=750)
        # fig.write_html(Path(self.folderName, "modalities.html"))

    def sunburstTasks(self):
        print("======================= Creating sunburst tasks =======================")
        import plotly.express as px

        title = "Task"
        totalSamples = 0

        # Create a dataframe with column "modality", "task", "dataset" and "size"
        data = []
        for dataset in self.datasets:
            data.append(
                {
                    "title": title,
                    "modality": dataset.modality,
                    "task": dataset.task,
                    "dataset": dataset.taskName,
                    "size": len(dataset),
                }
            )
            totalSamples += len(dataset)

        df = pd.DataFrame(columns=["title", "modality", "task", "dataset", "size"], data=data)

        fig = px.sunburst(df, path=["title", "task", "modality", "dataset"], values="size")
        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

        # fig.update_layout(title_text=f"Task (Total samples: {totalSamples})")
        fig.write_image(Path(self.folderName, "tasks.png"), scale=1.0, width=750, height=750)
        # fig.write_html(Path(self.folderName, "tasks.html"))

    def tableImageClassification(self):
        print("======================= Creating table image classification =======================")
        import plotly.graph_objects as go

        # Create a dataframe with column "modality", "task", "dataset" and "size"
        data = []
        for dataset in self.datasets:
            # Get scoringType if present else nan
            try:
                scoringType = dataset.scoringType
            except:
                scoringType = "NaN"

            data.append(
                {
                    "modality": dataset.modality,
                    "task": dataset.task,
                    "dataset": dataset.taskName,
                    "task type": scoringType,
                    "size": len(dataset),
                }
            )

        df = pd.DataFrame(columns=["modality", "task", "task type", "dataset", "size"], data=data)

        # Only keep the image classification task datasets
        df = df[df["task"] == "Image Classification"]

        # Only keep the columns "modality", "dataset" and "size"
        mdDf = df[["modality", "task type", "dataset", "size"]]
        # Order by modality
        mdDf = mdDf.sort_values(by=["modality"])

        # Conver to markdown and join the lines with the same modality
        mdDf = mdDf.to_markdown(index=False)
        print(mdDf)

        # Print a table showing the modality, dataset name and then size
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=["Modality", "Dataset", "Size"],
                        font=dict(size=14),
                        align="left",
                    ),
                    cells=dict(
                        values=[
                            df["modality"],
                            df["dataset"],
                            df["size"],
                        ],
                        align="left",
                    ),
                )
            ]
        )
        fig.update_layout(title_text="Image Classification")
        fig.write_image(Path(self.folderName, "image_classification.png"), scale=1.0, width=1920, height=1080)
        # fig.write_html(Path(self.folderName, "image_classification.html"))

    def sankeyDiagram(self):
        print("======================= Creating sankey diagram =======================")
        import plotly.graph_objects as go
        import plotly.express as px
        import random

        # The labels are the name of the datasets, the name of the tasks and the name of the modalities
        labelToIdx = {}
        tasks = set()
        modalities = set()
        datasetToLength = {}

        for dataset in self.datasets:
            # Add the dataset name to the labels
            if dataset.taskName not in labelToIdx:
                labelToIdx[dataset.taskName] = len(labelToIdx)
                datasetToLength[dataset.taskName] = len(dataset)

            # Add the task name to the labels
            if dataset.task not in labelToIdx:
                labelToIdx[dataset.task] = len(labelToIdx)
                tasks.add(dataset.task)

            # Add the modality name to the labels
            if dataset.modality not in labelToIdx:
                labelToIdx[dataset.modality] = len(labelToIdx)
                modalities.add(dataset.modality)
        


        # Create the colors: strong colors for the tasks and variations of the same color for the dataset with the same task

        # Take colors from the G10 palette
        indexToColor = {}
        for idx, task in enumerate(tasks):
            indexToColor[labelToIdx[task]] = idx / len(tasks)

        for dataset in self.datasets:
            # Sample a color based on the task of the dataset
            taskColor = indexToColor[labelToIdx[dataset.task]]

            # Sample small variation of the color for the dataset
            rangeColor = 1 / len(tasks) / 2
            datasetColor = (taskColor + random.uniform(-rangeColor, rangeColor)) % 1
            indexToColor[labelToIdx[dataset.taskName]] = datasetColor

        for idx, modality in enumerate(modalities):
            # Get all the tasks that have this modality
            taskColors = []
            taskWeights = []
            for dataset in self.datasets:
                if dataset.modality == modality:
                    taskColors.append(indexToColor[labelToIdx[dataset.task]])
                    taskWeights.append(len(dataset))

            # Convert angles to Cartesian coordinates
            x_coords = [math.cos(angle) for angle in taskColors]
            y_coords = [math.sin(angle) for angle in taskColors]

            # Calculate the weighted sums
            weighted_sum_x = sum(w * x for w, x in zip(taskWeights, x_coords))
            weighted_sum_y = sum(w * y for w, y in zip(taskWeights, y_coords))

            # Calculate the weighted mean angle
            weighted_mean_angle = math.atan2(weighted_sum_y, weighted_sum_x) % (2 * math.pi)

            indexToColor[labelToIdx[modality]] = weighted_mean_angle
        
        # Convert the colors to rgb
        for idx, color in indexToColor.items():
            indexToColor[idx] = px.colors.sample_colorscale("mrybm", color)[0]



        # Create the links
        source = []
        target = []
        value = []

        for dataset in self.datasets:
            # Add the link between the dataset and the task
            source.append(labelToIdx[dataset.taskName])
            target.append(labelToIdx[dataset.task])
            value.append(len(dataset))

            # Add the link between the task and the modality
            source.append(labelToIdx[dataset.task])
            target.append(labelToIdx[dataset.modality])
            value.append(len(dataset))

        labels = list(labelToIdx.keys())
        colors = [indexToColor[labelToIdx[label]] for label in labels]

        # Create the figure
        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=50,
                        line=dict(color="black", width=0.5),
                        label=labels,
                        color=colors,
                    ),
                    link=dict(
                        source=source,
                        target=target,
                        value=value,
                    ),
                )
            ]
        )

        for x_coordinate, column_name in enumerate(["Datasets", "Tasks", "Modalities"]):
            fig.add_annotation(
                x=x_coordinate,
                y=1.05,
                xref="x",
                yref="paper",
                text=column_name,
                showarrow=False,
                font=dict(family="Courier New, monospace", size=16, color="black"),
                align="center",
            )

        fig.update_layout(
            xaxis={
                "showgrid": False,  # thin lines in the background
                "zeroline": False,  # thick line at x=0
                "visible": False,  # numbers below
            },
            yaxis={
                "showgrid": False,  # thin lines in the background
                "zeroline": False,  # thick line at x=0
                "visible": False,  # numbers below
            },
            plot_bgcolor="rgba(0,0,0,0)",
            font_size=20,
        )

        # Reduce margins
        fig.update_layout(margin=dict(t=1, l=0.5, r=2, b=0))
        # Increase font size of the labels

        fig.write_image(Path(self.folderName, "sankey.png"), scale=1.0, width=1500, height=700)

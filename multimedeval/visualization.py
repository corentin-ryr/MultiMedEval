from multimedeval.utils import Benchmark
import pandas as pd
from pathlib import Path
import math


class BenchmarkVisualizer:
    def __init__(self, datasets: list[Benchmark]) -> None:
        self.datasets = datasets

        self.folderName = "figures"

        # Create the folder if it doesn't exist
        Path(self.folderName).mkdir(parents=True, exist_ok=True)

    def sunburstModalities(self):
        self._importPlotly()
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
        self._importPlotly()
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
        self._importPlotly()
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
        # df = df[df["task"] == "Image Classification"]

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
        self._importPlotly()
        print("======================= Creating sankey diagram =======================")
        import plotly.graph_objects as go
        import plotly.express as px
        import random
        from colour import Color

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


        indexToColor:dict[int, Color] = {}
        mainColors = ["#D6B656", "#6C8EBF", "#B85450", "#82B366", "#9673A6", "#D79B00"]
        for idx, task in enumerate(tasks):
            indexToColor[labelToIdx[task]] = Color(mainColors[idx])

        for dataset in self.datasets:
            # Sample a color based on the task of the dataset
            taskColor:Color = indexToColor[labelToIdx[dataset.task]]

            # Sample small variation of the color for the dataset
            rangeColor = 1 / len(tasks) / 4
            datasetColor = Color(taskColor, hue=taskColor.get_hue() + random.uniform(-rangeColor, rangeColor) % 1)
            indexToColor[labelToIdx[dataset.taskName]] = datasetColor

        for idx, modality in enumerate(modalities):
            # Get all the tasks that have this modality
            taskColors = []
            taskWeights = []
            baseColor = None
            for dataset in self.datasets:
                if dataset.modality == modality:
                    taskColors.append(indexToColor[labelToIdx[dataset.task]].get_hue())
                    taskWeights.append(len(dataset))
                    baseColor = indexToColor[labelToIdx[dataset.task]]

            weighted_mean_hue = self._averageCircular(taskColors, taskWeights)
            indexToColor[labelToIdx[modality]] = Color(baseColor, hue=weighted_mean_hue)

        # Create the links
        source = []
        target = []
        value = []
        linksColor = []

        for dataset in self.datasets:
            # Add the link between the dataset and the task
            source.append(labelToIdx[dataset.taskName])
            target.append(labelToIdx[dataset.task])
            value.append(len(dataset))
            linksColor.append(self._averageCircular([indexToColor[labelToIdx[dataset.taskName]].get_hue(), indexToColor[labelToIdx[dataset.task]].get_hue()]))

            # Add the link between the task and the modality
            source.append(labelToIdx[dataset.task])
            target.append(labelToIdx[dataset.modality])
            value.append(len(dataset))
            linksColor.append(self._averageCircular([indexToColor[labelToIdx[dataset.task]].get_hue(), indexToColor[labelToIdx[dataset.modality]].get_hue()]))

        for idx in range(len(linksColor)):
            tempColor = Color(hsl=(linksColor[idx], 0.5, 0.7)).rgb
            linksColor[idx] = f"rgb({tempColor[0] * 255}, {tempColor[1] * 255}, {tempColor[2] * 255})"

        labels = list(labelToIdx.keys())
        colors = [indexToColor[labelToIdx[label]].rgb for label in labels]
        colors = [f"rgb({color[0] * 255}, {color[1] * 255}, {color[2] * 255})" for color in colors]

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
                        color=linksColor,
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
                font=dict(family="Helvetica", size=20, color="black"),
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
            font_size=16,
            font_family="Helvetica",
        )

        # Reduce margins
        fig.update_layout(margin=dict(t=50, l=20, r=20, b=20))
        # Increase font size of the labels

        fig.write_image(Path(self.folderName, "sankey.png"), scale=1.0, width=1500, height=700)

    def sankeyD3Blocks(self):
        from d3blocks import D3Blocks
        import random
        from colour import Color

        print("======================= Creating sankey diagram with D3Blocks =======================")

        dfAsList = []
        tasks = set()
        modalities = set()

        for dataset in self.datasets:
            task = dataset.task.replace("_", " ")
            # Add the link between the dataset and the task
            dfAsList.append((dataset.taskName, task, len(dataset)))

            # Add the link between the task and the modality
            dfAsList.append((task, dataset.modality, len(dataset)))

            tasks.add(task)
            modalities.add(dataset.modality)

        # Take colors from the G10 palette
        nameToColor = {}
        for idx, task in enumerate(tasks):
            nameToColor[task] = (idx / len(tasks) + 0.5) % 1

        for dataset in self.datasets:
            # Sample a color based on the task of the dataset
            taskColor = nameToColor[dataset.task.replace("_", " ")]

            # Sample small variation of the color for the dataset
            rangeColor = 1 / len(tasks) / 4
            datasetColor = (taskColor + random.uniform(-rangeColor, rangeColor)) % 1
            nameToColor[dataset.taskName] = datasetColor

        for idx, modality in enumerate(modalities):
            # Get all the tasks that have this modality
            taskColors = []
            taskWeights = []
            for dataset in self.datasets:
                if dataset.modality == modality:
                    taskColors.append(nameToColor[dataset.task])
                    taskWeights.append(len(dataset))

            weighted_mean_angle = self._averageCircular(taskColors, taskWeights)

            nameToColor[modality] = weighted_mean_angle

        # Convert every color to hex
        for name, color in nameToColor.items():
            tempColor = Color(hsl=(color, 0.5, 0.4)).hex
            nameToColor[name] = tempColor

        df = pd.DataFrame(dfAsList, columns=["source", "target", "weight"])
        print(df)

        d3 = D3Blocks(chart="Sankey", frame=True)
        
        # Change the font
        d3.set_node_properties(df, width=30, color=nameToColor)

        d3.set_edge_properties(df, color="source-target", opacity=0.8)

        d3.show(filepath="tempSankey.html", figsize=(1000, 700))

    def _importPlotly(self):
        try:
            import plotly
            import kaleido
            import tabulate
        except ImportError:
            print("Please install plotly and kaleido with `pip install plotly kaleido tabulate` to generate the visualizations.")
            
            exit(1)


    def _averageCircular(self, angles, weights=None):
        # Convert angles to Cartesian coordinates
        x_coords = [math.cos(angle * 2 * math.pi) for angle in angles]
        y_coords = [math.sin(angle * 2 * math.pi) for angle in angles]

        # Calculate the weighted sums
        if weights is None:
            weights = [1] * len(x_coords)
            
        weighted_sum_x = sum(w * x for w, x in zip(weights, x_coords))
        weighted_sum_y = sum(w * y for w, y in zip(weights, y_coords))

        # Calculate the weighted mean angle
        weightedAngle = math.atan2(weighted_sum_y, weighted_sum_x) % (2 * math.pi)

        # Convert to [0, 1] range
        return weightedAngle / (2 * math.pi)

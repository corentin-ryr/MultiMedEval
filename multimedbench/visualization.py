from multimedbench.utils import Benchmark
import pandas as pd
from pathlib import Path


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
        fig.update_layout(title_text=f"Modality (Total samples: {totalSamples})")
        fig.write_image(Path(self.folderName, "modalities.png"), scale=1.0, width=1920, height=1080)
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
        fig.update_layout(title_text=f"Task (Total samples: {totalSamples})")
        fig.write_image(Path(self.folderName, "tasks.png"), scale=1.0, width=1920, height=1080)
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

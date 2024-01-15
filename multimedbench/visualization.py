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
        fig.write_html(Path(self.folderName, "modalities.html"))


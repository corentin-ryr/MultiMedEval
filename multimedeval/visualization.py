"""Module to visualize the benchmarks."""

import math
import random
import sys
from pathlib import Path
from typing import List

import pandas as pd

from multimedeval.utils import Benchmark


class BenchmarkVisualizer:
    """Class to visualize the benchmarks."""

    def __init__(self, datasets: List[Benchmark]) -> None:
        """Initialize the BenchmarkVisualizer object."""
        self.datasets = datasets

        self.folder_name = "visualizations"

        # Create the folder if it doesn't exist
        Path(self.folder_name).mkdir(parents=True, exist_ok=True)

    def sunburst_modalities(self):
        """Create a sunburst plot showing the relationship between the \
            modalities and the datasets."""
        self._import_plotly()
        print(
            "======================= Creating sunburst modalities ======================="
        )
        import plotly.express as px  # pylint: disable=import-outside-toplevel

        title = "Modality"
        total_samples = 0

        # Create a dataframe with column "modality", "task", "dataset" and "size"
        data = []
        for dataset in self.datasets:
            data.append(
                {
                    "title": title,
                    "modality": dataset.modality,
                    "task": dataset.task,
                    "dataset": dataset.task_name,
                    "size": len(dataset),
                }
            )
            total_samples += len(dataset)
        df = pd.DataFrame(
            columns=["title", "modality", "task", "dataset", "size"], data=data
        )

        fig = px.sunburst(
            df, path=["title", "modality", "task", "dataset"], values="size"
        )
        fig.update_layout(margin={"t": 0, "l": 0, "r": 0, "b": 0})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

        # fig.update_layout(title_text=f"Modality (Total samples: {totalSamples})")
        fig.write_image(
            Path(self.folder_name, "modalities.png"), scale=1.0, width=750, height=750
        )
        # fig.write_html(Path(self.folderName, "modalities.html"))

    def sunburst_tasks(self):
        """Create a sunburst plot showing the relationship between the tasks and the datasets."""
        self._import_plotly()
        print("======================= Creating sunburst tasks =======================")
        import plotly.express as px  # pylint: disable=import-outside-toplevel

        title = "Task"
        total_samples = 0

        # Create a dataframe with column "modality", "task", "dataset" and "size"
        data = []
        for dataset in self.datasets:
            data.append(
                {
                    "title": title,
                    "modality": dataset.modality,
                    "task": dataset.task,
                    "dataset": dataset.task_name,
                    "size": len(dataset),
                }
            )
            total_samples += len(dataset)

        df = pd.DataFrame(
            columns=["title", "modality", "task", "dataset", "size"], data=data
        )

        fig = px.sunburst(
            df, path=["title", "task", "modality", "dataset"], values="size"
        )
        fig.update_layout(margin={"t": 0, "l": 0, "r": 0, "b": 0})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

        # fig.update_layout(title_text=f"Task (Total samples: {totalSamples})")
        fig.write_image(
            Path(self.folder_name, "tasks.png"), scale=1.0, width=750, height=750
        )
        # fig.write_html(Path(self.folderName, "tasks.html"))

    def table_image_classification(self):
        """Create a table showing the image classification datasets."""
        self._import_plotly()
        print(
            "======================= Creating table image classification ======================="
        )
        import plotly.graph_objects as go  # pylint: disable=import-outside-toplevel

        # Create a dataframe with column "modality", "task", "dataset" and "size"
        data = []
        for dataset in self.datasets:
            # Get scoringType if present else nan
            try:
                scoring_type = dataset.scoringType
            except AttributeError:
                scoring_type = "NaN"

            data.append(
                {
                    "modality": dataset.modality,
                    "task": dataset.task,
                    "dataset": dataset.task_name,
                    "task type": scoring_type,
                    "size": len(dataset),
                }
            )

        df = pd.DataFrame(
            columns=["modality", "task", "task type", "dataset", "size"], data=data
        )

        # Only keep the image classification task datasets
        # df = df[df["task"] == "Image Classification"]

        # Only keep the columns "modality", "dataset" and "size"
        md_df = df[["modality", "task type", "dataset", "size"]]
        # Order by modality
        md_df = md_df.sort_values(by=["modality"])

        # Conver to markdown and join the lines with the same modality
        md_df = md_df.to_markdown(index=False)
        print(md_df)

        # Print a table showing the modality, dataset name and then size
        fig = go.Figure(
            data=[
                go.Table(
                    header={
                        "values": ["Modality", "Dataset", "Size"],
                        "font": {"size": 14},
                        "align": "left",
                    },
                    cells={
                        "values": [
                            df["modality"],
                            df["dataset"],
                            df["size"],
                        ],
                        "align": "left",
                    },
                )
            ]
        )
        fig.update_layout(title_text="Image Classification")
        fig.write_image(
            Path(self.folder_name, "image_classification.png"),
            scale=1.0,
            width=1920,
            height=1080,
        )
        # fig.write_html(Path(self.folderName, "image_classification.html"))

    def sankey_diagram(self):
        """Create a sankey diagram showing the relationship between the datasets,
        tasks and modalities."""
        self._import_plotly()
        print("======================= Creating sankey diagram =======================")

        import plotly.graph_objects as go  # pylint: disable=import-outside-toplevel
        from colour import Color  # pylint: disable=import-outside-toplevel

        # The labels are the name of the datasets,
        # the name of the tasks and the name of the modalities
        label_to_idx = {}
        tasks = set()
        modalities = set()
        dataset_to_length = {}

        for dataset in self.datasets:
            # Add the dataset name to the labels
            if dataset.task_name not in label_to_idx:
                label_to_idx[dataset.task_name] = len(label_to_idx)
                dataset_to_length[dataset.task_name] = len(dataset)

            # Add the task name to the labels
            if dataset.task not in label_to_idx:
                label_to_idx[dataset.task] = len(label_to_idx)
                tasks.add(dataset.task)

            # Add the modality name to the labels
            if dataset.modality not in label_to_idx:
                label_to_idx[dataset.modality] = len(label_to_idx)
                modalities.add(dataset.modality)

        index_to_color: dict[int, Color] = {}
        main_colors = ["#D6B656", "#6C8EBF", "#B85450", "#82B366", "#9673A6", "#D79B00"]
        for idx, task in enumerate(tasks):
            index_to_color[label_to_idx[task]] = Color(main_colors[idx])

        for dataset in self.datasets:
            # Sample a color based on the task of the dataset
            task_color: Color = index_to_color[label_to_idx[dataset.task]]

            # Sample small variation of the color for the dataset
            range_color = 1 / len(tasks) / 4
            dataset_color = Color(
                task_color,
                hue=task_color.get_hue()
                + random.uniform(-range_color, range_color) % 1,
            )
            index_to_color[label_to_idx[dataset.task_name]] = dataset_color

        for modality in modalities:
            # Get all the tasks that have this modality
            task_colors = []
            tas_weights = []
            base_color = None
            for dataset in self.datasets:
                if dataset.modality == modality:
                    task_colors.append(
                        index_to_color[label_to_idx[dataset.task]].get_hue()
                    )
                    tas_weights.append(len(dataset))
                    base_color = index_to_color[label_to_idx[dataset.task]]

            weighted_mean_hue = self._average_circular(task_colors, tas_weights)
            index_to_color[label_to_idx[modality]] = Color(
                base_color, hue=weighted_mean_hue
            )

        # Create the links
        source = []
        target = []
        value = []
        links_color = []

        for dataset in self.datasets:
            # Add the link between the dataset and the task
            source.append(label_to_idx[dataset.task_name])
            target.append(label_to_idx[dataset.task])
            value.append(len(dataset))
            links_color.append(
                self._average_circular(
                    [
                        index_to_color[label_to_idx[dataset.task_name]].get_hue(),
                        index_to_color[label_to_idx[dataset.task]].get_hue(),
                    ]
                )
            )

            # Add the link between the task and the modality
            source.append(label_to_idx[dataset.task])
            target.append(label_to_idx[dataset.modality])
            value.append(len(dataset))
            links_color.append(
                self._average_circular(
                    [
                        index_to_color[label_to_idx[dataset.task]].get_hue(),
                        index_to_color[label_to_idx[dataset.modality]].get_hue(),
                    ]
                )
            )

        for idx, link_idx in enumerate(links_color):
            temp_color = Color(hsl=(link_idx, 0.5, 0.7)).rgb
            links_color[idx] = (
                f"rgb({temp_color[0] * 255}, {temp_color[1] * 255}, {temp_color[2] * 255})"
            )

        labels = list(label_to_idx.keys())
        colors = [index_to_color[label_to_idx[label]].rgb for label in labels]
        colors = [
            f"rgb({color[0] * 255}, {color[1] * 255}, {color[2] * 255})"
            for color in colors
        ]

        # Create the figure
        fig = go.Figure(
            data=[
                go.Sankey(
                    node={
                        "pad": 15,
                        "thickness": 50,
                        "line": {"color": "black", "width": 0.5},
                        "label": labels,
                        "color": colors,
                    },
                    link={
                        "source": source,
                        "target": target,
                        "value": value,
                        "color": links_color,
                    },
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
                font={"family": "Helvetica", "size": 25, "color": "black"},
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
            font_family="Helvetica",
        )

        # Reduce margins
        fig.update_layout(margin={"t": 50, "l": 20, "r": 20, "b": 20})
        # Increase font size of the labels

        fig.write_image(
            Path(self.folder_name, "sankey.png"), scale=1.0, width=1500, height=700
        )

    def _import_plotly(self):
        try:
            import kaleido  # pylint: disable=unused-import, import-outside-toplevel # noqa
            import plotly  # pylint: disable=unused-import, import-outside-toplevel # noqa
            import tabulate  # pylint: disable=unused-import, import-outside-toplevel # noqa
        except ImportError:
            print(
                "Please install plotly and kaleido with "
                "`pip install plotly kaleido tabulate` to generate the visualizations."
            )

            sys.exit(1)

    def _average_circular(self, angles, weights=None):
        # Convert angles to Cartesian coordinates
        x_coords = [math.cos(angle * 2 * math.pi) for angle in angles]
        y_coords = [math.sin(angle * 2 * math.pi) for angle in angles]

        # Calculate the weighted sums
        if weights is None:
            weights = [1] * len(x_coords)

        weighted_sum_x = sum(w * x for w, x in zip(weights, x_coords))
        weighted_sum_y = sum(w * y for w, y in zip(weights, y_coords))

        # Calculate the weighted mean angle
        weighted_angle = math.atan2(weighted_sum_y, weighted_sum_x) % (2 * math.pi)

        # Convert to [0, 1] range
        return weighted_angle / (2 * math.pi)

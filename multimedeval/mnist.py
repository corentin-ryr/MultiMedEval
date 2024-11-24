"""The MNIST dataset family."""

import os
from typing import List, Optional, Union

from medmnist.dataset import (
    OCTMNIST,
    BloodMNIST,
    BreastMNIST,
    ChestMNIST,
    DermaMNIST,
    OrganAMNIST,
    OrganCMNIST,
    OrganSMNIST,
    PathMNIST,
    PneumoniaMNIST,
    RetinaMNIST,
    TissueMNIST,
)

from multimedeval.task_families import ImageClassification
from multimedeval.utils import clean_str, BatcherInput

NAME_TO_MNIST = {
    "OCTMNIST": {"class": OCTMNIST, "modality": "OCT"},
    "PathMNIST": {"class": PathMNIST, "modality": "Pathology"},
    "PneumoniaMNIST": {"class": PneumoniaMNIST, "modality": "X-Ray"},
    "RetinaMNIST": {"class": RetinaMNIST, "modality": "Fundus Camera"},
    "BloodMNIST": {"class": BloodMNIST, "modality": "Microscope"},
    "ChestMNIST": {"class": ChestMNIST, "modality": "X-Ray"},
    "OrganAMNIST": {"class": OrganAMNIST, "modality": "CT"},
    "OrganCMNIST": {"class": OrganCMNIST, "modality": "CT"},
    "OrganSMNIST": {"class": OrganSMNIST, "modality": "CT"},
    "DermaMNIST": {"class": DermaMNIST, "modality": "Dermatology"},
    "BreastMNIST": {"class": BreastMNIST, "modality": "Ultrasound"},
    "TissueMNIST": {"class": TissueMNIST, "modality": "Microscope"},
}


class _wrapperGenerator:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        sample = self.dataset[index]
        return {"image": sample[0], "label": sample[1]}

    def __len__(self):
        return len(self.dataset)


class MNIST(ImageClassification):
    """The MNIST dataset family."""

    def __init__(self, mnist_name, **kwargs) -> None:
        """Initialize the MNIST dataset family."""
        super().__init__(**kwargs)
        self.task_name = mnist_name
        self.modality = NAME_TO_MNIST[mnist_name]["modality"]

        self.question = ""
        self.cache_dir_name: str = ""
        self.cache_dir: Optional[os.PathLike] = None

    def setup(self):
        """Setup the MNIST dataset family."""
        self.cache_dir = self.engine.get_config()[self.cache_dir_name]

        if self.cache_dir is None:
            raise ValueError(
                f"Skipping {self.task_name} because the cache directory is not set."
            )

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        self.dataset = NAME_TO_MNIST[self.task_name]["class"](
            split="test", download=True, root=self.cache_dir
        )
        self.options: dict[str, str] = self.dataset.info["label"]
        # Add 1 to the key of the options
        self.options = {str(int(key) + 1): value for key, value in self.options.items()}

        self.num_classes = len(self.options)
        self.scoring_type = self.dataset.info["task"].split(",")[0].replace("-", "")
        if self.scoring_type in ["binaryclass", "ordinalregression"]:
            self.scoring_type = "multiclass"

        self.dataset = _wrapperGenerator(self.dataset)

        self.train_dataset = NAME_TO_MNIST[self.task_name]["class"](
            split="train", download=True, root=self.cache_dir
        )
        self.train_dataset = _wrapperGenerator(self.train_dataset)

    def get_correct_answer(self, sample, full_text=False) -> Union[int, str, List[int]]:
        """Returns the correct answer for the sample.

        Args:
            sample: The sample to get the correct answer from.
            fullText: Whether to return the full answer. Defaults to False.

        Returns:
            The correct answer
        """
        label = sample["label"].tolist()

        if full_text:
            return ",".join([self.options[str(label + 1)] for label in label])

        if len(label) == 1:
            label = label[0]

        return label

    def format_question(self, sample, prompt=False):
        """Formats the question.

        Args:
            sample: The sample to format.
            prompt: Adds the answer to the prompt. Defaults to False.

        Returns:
            An instance of BatcherInput with the formatted prompt and the images.
        """
        question = "<img> Options:\n"
        question += " \n ".join(
            [f"{option}: {self.options[option]}" for option in self.options]
        )
        question += " \n Which options correspond to the image?"

        # formatted_text = [
        #     {
        #         "role": "user",
        #         "content": question,
        #     }
        # ]
        batcher_input = BatcherInput()
        batcher_input._add_text_prompt('user', question)
        if prompt:
            # formatted_text.append(
            #     {
            #         "role": "assistant",
            #         "content": f"{self.get_correct_answer(sample, full_text=True)}",
            #     }
            # )
            batcher_input._add_text_prompt('assistant', f"{self.get_correct_answer(sample, full_text=True)}")
        batcher_input._add_images([sample["image"]])
        return batcher_input

    def get_predicted_answer(self, answer) -> Union[int, List[int]]:
        """Converts the free form text output to the answer index.

        Args:
            sample: The sample used to generate the answer
            answer: The free form text output of the model

        Returns:
            The index of the answer
        """
        answer = clean_str(answer)
        # Find the best bleu score between the answer and the options
        options = [
            clean_str(f"{option}: {self.options[option]}") for option in self.options
        ]
        scores = [self.bleu([answer], [[option]]) for option in options]

        if self.scoring_type == "multiclass":
            return scores.index(max(scores))

        # for each 1 if above a threshold, 0 otherwise
        return [1 if score > 0.5 else 0 for score in scores]


class MNISTOct(MNIST):
    """The OCTMNIST dataset."""

    def __init__(self, **kwargs) -> None:
        """Initialize the OCTMNIST dataset."""
        super().__init__("OCTMNIST", **kwargs)
        self.question = "Diagnose this retina OCT."
        self.cache_dir_name = "mnist_oct_dir"


class MNISTPath(MNIST):
    """The PathMNIST dataset."""

    def __init__(self, **kwargs) -> None:
        """Initialize the PathMNIST dataset."""
        super().__init__("PathMNIST", **kwargs)
        self.question = "Which kind of tissue is represented in the image?"
        self.cache_dir_name = "mnist_path_dir"


class MNISTPneumonia(MNIST):
    """The PneumoniaMNIST dataset."""

    def __init__(self, **kwargs) -> None:
        """Initialize the PneumoniaMNIST dataset."""
        super().__init__("PneumoniaMNIST", **kwargs)
        self.question = "Diagnose this chest X-Ray."
        self.cache_dir_name = "mnist_pneumonia_dir"


class MNISTRetina(MNIST):
    """The RetinaMNIST dataset."""

    def __init__(self, **kwargs) -> None:
        """Initialize the RetinaMNIST dataset."""
        super().__init__("RetinaMNIST", **kwargs)
        self.question = (
            "Grade this diabetic retinopathy following the "
            "international clinical DR severity scale."
        )
        self.cache_dir_name = "mnist_retina_dir"


class MNISTBlood(MNIST):
    """The BloodMNIST dataset."""

    def __init__(self, **kwargs) -> None:
        """Initialize the BloodMNIST dataset."""
        super().__init__("BloodMNIST", **kwargs)
        self.question = "What kind of peripheral blood cells is shown in the image?"
        self.cache_dir_name = "mnist_blood_dir"


# class MNIST_Chest(MNIST):
#     def __init__(self, **kwargs) -> None:
#         super().__init__("ChestMNIST", **kwargs)
#         self.question = "Which thoracic diseases can be seen in this chest X-Ray?"
#         self.cache_dir_name = "MedMNIST_dir"]


# class MNIST_OrganA(MNIST):
#     def __init__(self, **kwargs) -> None:
#         super().__init__("OrganAMNIST", **kwargs)
#         self.question = "Which organ is present in the image?"
#         self.cache_dir_name = "MedMNIST_dir"]


class MNISTOrganC(MNIST):
    """The OrganCMNIST dataset."""

    def __init__(self, **kwargs) -> None:
        """Initialize the OrganCMNIST dataset."""
        super().__init__("OrganCMNIST", **kwargs)
        self.question = "Which organ is present in the image?"
        self.cache_dir_name = "mnist_organc_dir"


class MNISTDerma(MNIST):
    """The DermaMNIST dataset."""

    def __init__(self, **kwargs) -> None:
        """Initialize the DermaMNIST dataset."""
        super().__init__("DermaMNIST", **kwargs)
        self.question = "Which skin disease is present in the image?"
        self.cache_dir_name = "mnist_derma_dir"


class MNISTBreast(MNIST):
    """The BreastMNIST dataset."""

    def __init__(self, **kwargs) -> None:
        """Initialize the BreastMNIST dataset."""
        super().__init__("BreastMNIST", **kwargs)
        self.question = (
            "Does this breast ultrasound show sign of malignant tumor or is it benign?"
        )
        self.cache_dir_name = "mnist_breast_dir"


class MNISTTissue(MNIST):
    """The TissueMNIST dataset."""

    def __init__(self, **kwargs) -> None:
        """Initialize the TissueMNIST dataset."""
        super().__init__("TissueMNIST", **kwargs)
        self.question = "What kind of tissue is represented in the image?"
        self.cache_dir_name = "mnist_tissue_dir"

    def __len__(self):
        """Returns the length of the dataset."""
        return super().__len__() // 4


class MNISTOrganS(MNIST):
    """The OrganSMNIST dataset."""

    def __init__(self, **kwargs) -> None:
        """Initialize the OrganSMNIST dataset."""
        super().__init__("OrganSMNIST", **kwargs)
        self.question = "Which organ is present in the image?"
        self.cache_dir_name = "mnist_organs_dir"

from multimedeval.taskFamilies import ImageClassification
from medmnist.dataset import OCTMNIST, PathMNIST, PneumoniaMNIST, RetinaMNIST, BloodMNIST, ChestMNIST, OrganAMNIST, OrganCMNIST, DermaMNIST, BreastMNIST, TissueMNIST, OrganSMNIST, MedMNIST2D
import os
from multimedeval.utils import cleanStr

NAME_TO_MNIST = {
    "OCTMNIST": {"class": OCTMNIST, "modality": "OCT" },
    "PathMNIST": {"class": PathMNIST, "modality": "Pathology" },
    "PneumoniaMNIST": {"class": PneumoniaMNIST, "modality": "X-Ray" },
    "RetinaMNIST": {"class": RetinaMNIST, "modality": "Fundus Camera" },
    "BloodMNIST": {"class": BloodMNIST, "modality": "Microscope" },
    "ChestMNIST": {"class": ChestMNIST, "modality": "X-Ray" },
    "OrganAMNIST": {"class": OrganAMNIST, "modality": "CT" },
    "OrganCMNIST": {"class": OrganCMNIST, "modality": "CT" },
    "OrganSMNIST": {"class": OrganSMNIST, "modality": "CT" },
    "DermaMNIST": {"class": DermaMNIST, "modality": "Dermatology" },
    "BreastMNIST": {"class": BreastMNIST, "modality": "Ultrasound" },
    "TissueMNIST": {"class": TissueMNIST, "modality": "Microscope" },
}

class wrapperGenerator():
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
            sample = self.dataset[index]
            return {"image": sample[0], "label": sample[1]}
    
    def __len__(self):
        return len(self.dataset)



class MNIST(ImageClassification):
    def __init__(self, mnistName, **kwargs) -> None:
        super().__init__(**kwargs)
        self.taskName = mnistName
        self.modality = NAME_TO_MNIST[mnistName]["modality"]

        self.question = None

    def setup(self):
        self.cacheDir = self.engine.getConfig()[self.cachedirName]

        if self.cacheDir is None:
            raise Exception(f"Skipping {self.taskName} because the cache directory is not set.")

        if not os.path.exists(self.cacheDir):
            os.makedirs(self.cacheDir, exist_ok=True)

        self.dataset = NAME_TO_MNIST[self.taskName]["class"](split="test", download=True, root=self.cacheDir)
        self.options:dict[str, str] = self.dataset.info["label"]
        # Add 1 to the key of the options
        self.options = {str(int(key) + 1): value for key, value in self.options.items()}

        self.num_classes = len(self.options)
        self.scoringType = self.dataset.info["task"].split(",")[0].replace("-", "")
        if self.scoringType in ["binaryclass", "ordinalregression"]:
            self.scoringType = "multiclass"
        
        self.dataset = wrapperGenerator(self.dataset)

        self.trainDataset = NAME_TO_MNIST[self.taskName]["class"](split="train", download=True, root=self.cacheDir)
        self.trainDataset = wrapperGenerator(self.trainDataset)

    def getCorrectAnswer(self, sample, fullText=False) -> int:
        label = sample["label"].tolist()
        
        if fullText:
            return ",".join([self.options[str(label + 1)] for label in label])
        
        if len(label) == 1:
            label = label[0]
        
        return label

    def format_question(self, sample, prompt=False):
        question = "<img> Options:\n"
        question += " \n ".join([f"{option}: {self.options[option]}" for option in self.options])
        question += " \n Which options correspond to the image?"

        formattedText = [
            {
                "role": "user",
                "content": question,
            }
        ]
        if prompt:
            formattedText.append({"role": "assistant", "content": f"{self.getCorrectAnswer(sample, fullText=True)}"})

        return (formattedText, [sample["image"]])

    def getPredictedAnswer(self, answer) -> int:
        """Converts the free form text output to the answer index

        Args:
            sample (_type_): The sample used to generate the answer
            answer (_type_): The free form text output of the model

        Returns:
            int: The index of the answer
        """
        answer = cleanStr(answer)
        # Find the best bleu score between the answer and the options
        options = [cleanStr(f"{option}: {self.options[option]}") for option in self.options]
        scores = [self.bleu([answer], [[option]]) for option in options]

        if self.scoringType == "multiclass":
            return scores.index(max(scores))
        else:
            # for each 1 if above a threshold, 0 otherwise
            return [1 if score > 0.5 else 0 for score in scores]



class MNIST_Oct(MNIST):
    def __init__(self, **kwargs) -> None:
        super().__init__("OCTMNIST", **kwargs)
        self.question = "Diagnose this retina OCT."
        self.cachedirName = "MNIST_Oct_dir"


class MNIST_Path(MNIST):
    def __init__(self, **kwargs) -> None:
        super().__init__("PathMNIST", **kwargs)
        self.question = "Which kind of tissue is represented in the image?"
        self.cachedirName = "MNIST_Path_dir"


class MNIST_Pneumonia(MNIST):
    def __init__(self, **kwargs) -> None:
        super().__init__("PneumoniaMNIST", **kwargs)
        self.question = "Diagnose this chest X-Ray."
        self.cachedirName = "MNIST_Pneumonia_dir"


class MNIST_Retina(MNIST):
    def __init__(self, **kwargs) -> None:
        super().__init__("RetinaMNIST", **kwargs)
        self.question = "Grade this diabetic retinopathy following the international clinical DR severity scale."
        self.cachedirName = "MNIST_Retina_dir"


class MNIST_Blood(MNIST):
    def __init__(self, **kwargs) -> None:
        super().__init__("BloodMNIST", **kwargs)
        self.question = "What kind of peripheral blood cells is shown in the image?"
        self.cachedirName = "MNIST_Blood_dir"


# class MNIST_Chest(MNIST):
#     def __init__(self, **kwargs) -> None:
#         super().__init__("ChestMNIST", **kwargs)
#         self.question = "Which thoracic diseases can be seen in this chest X-Ray?"
#         self.cachedirName = "MedMNIST_dir"]


# class MNIST_OrganA(MNIST):
#     def __init__(self, **kwargs) -> None:
#         super().__init__("OrganAMNIST", **kwargs)
#         self.question = "Which organ is present in the image?"
#         self.cachedirName = "MedMNIST_dir"]

class MNIST_OrganC(MNIST):
    def __init__(self, **kwargs) -> None:
        super().__init__("OrganCMNIST", **kwargs)
        self.question = "Which organ is present in the image?"
        self.cachedirName = "MNIST_OrganC_dir"

class MNIST_Derma(MNIST):
    def __init__(self, **kwargs) -> None:
        super().__init__("DermaMNIST", **kwargs)
        self.question = "Which skin disease is present in the image?"
        self.cachedirName = "MNIST_Derma_dir"

class MNIST_Breast(MNIST):
    def __init__(self, **kwargs) -> None:
        super().__init__("BreastMNIST", **kwargs)
        self.question = "Does this breast ultrasound show sign of malignant tumor or is it benign?"
        self.cachedirName = "MNIST_Breast_dir"

class MNIST_Tissue(MNIST):
    def __init__(self, **kwargs) -> None:
        super().__init__("TissueMNIST", **kwargs)
        self.question = "What kind of tissue is represented in the image?"
        self.cachedirName = "MNIST_Tissue_dir"
    
    def __len__(self):
        return super().__len__() // 4

class MNIST_OrganS(MNIST):
    def __init__(self, **kwargs) -> None:
        super().__init__("OrganSMNIST", **kwargs)
        self.question = "Which organ is present in the image?"
        self.cachedirName = "MNIST_OrganS_dir"


    
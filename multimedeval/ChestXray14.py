import os
import datasets
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image

from multimedeval.task_families import ImageClassification
from multimedeval.utils import clean_str, download_file
from glob import glob

class ChestXray14(ImageClassification):
    """ChestXray14 Image Classification task."""

    def __init__(self, **kwargs):
        """Initialize the ChestXray14 Image Classification task."""
        super().__init__(**kwargs)
        self.modality = "X-Ray"
        self.task_name = "ChestXray14"

    def setup(self):
        """Setup the ChestXray14 Image Classification task."""
        self.num_classes = 14
        self.scoring_type = "multilabel" #not sure, but multi-class doesn't work
        self.options = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax",
"Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]

        # Get the dataset from Kaggle
        self.path = self.engine.get_config()["chestxray14_dir"]

        if self.path is None:
            raise ValueError(
                "Skipping ChestXray14 because the cache directory is not set."
            )

        self._generate_dataset()

        # After the dataset generation
        full_dataset = pd.read_csv(self.path + 'Data_Entry_2017.csv')

        # map the image path as a new column in dataframe
        my_glob = glob(self.path + 'images*/images/*.png')
        full_img_paths = {os.path.basename(x): x for x in my_glob}
        full_dataset['full_path'] = full_dataset['Image Index'].map(full_img_paths.get)

        # Replace backslashes with forward slashes in the 'file_paths' column
        full_dataset['full_path'] = full_dataset['full_path'].str.replace("\\", "/")

        self.dataset, self.train_dataset = self._train_test_split(full_dataset)

    def get_predicted_answer(self, answer):
        """Convert the free form text output to the answer index.

        Args:
            answer: The free form text output of the model.

        Returns:
            A list of predicted indices of the answer, e.g. [1,1,0,...,0].
        """
        answer = clean_str(answer)
        # Find the best bleu score between the answer and the options
        scores = [self.bleu([answer], [[clean_str(option)]]) for option in self.options]

        # for each 1 if above a threshold, 0 otherwise,
        return [1 if score > 0.5 else 0 for score in scores]

    def get_correct_answer(self, sample, full_text=False):
        """Returns the indices of correct answer for the sample.

        Args:
            sample: The sample to get the correct answer from. 'label' may be a string with multiple labels separated by '|'.
            full_text: Whether to return the full text of the answer. Defaults to False.

        Returns:
            The correct answer as a list of true indices or a full-text string.
        """

        label = sample["Finding Labels"]
        # Split by '|' if necessary
        if isinstance(label, str) and '|' in label:
            label = label.split('|')
        else:
            label = [label] if isinstance(label, (int, str)) else label


        if full_text:
            # Return the full text for each label
            return ",".join([str(lbl) for lbl in label])

        label_presence = [0] * len(self.options)

        for lbl in label:
            if lbl in self.options:
                idx = self.options.index(lbl)
                label_presence[idx] =  1

        return label_presence

    def format_question(self, sample, prompt=False):
        """Formats the question.

        Args:
            sample: The sample to format.
            prompt: Adds the answer to the prompt. Defaults to False.

        Returns:
            A tuple with the formatted prompt and the images.
        """
        question = "<img> Options:\n"
        question += " \n ".join(
            [f"{option}" for option in self.options]
        )
        question += " \n List the options that can be seen in this picture."

        formatted_text = [
            {
                "role": "user",
                "content": question,
            }
        ]
        if prompt:
            formatted_text.append(
                {
                    "role": "assistant",
                    "content": f"{self.get_correct_answer(sample, full_text=True)}",
                }
            )

        image = Image.open(sample['full_path'])

        return (formatted_text, image)

    def _train_test_split(self, dataset):
        '''
            Split the dataset according to dataset documents.
        '''
        with open(self.path + 'train_val_list.txt', 'r') as file:
            # Read each line into a list and remove any trailing newlines
            train_list = [line.strip() for line in file]
            train_dataset = dataset[dataset['Image Index'].isin(train_list)]

        with open(self.path + 'test_list.txt', 'r') as file:
            test_list = [line.strip() for line in file]
            test_dataset = dataset[dataset['Image Index'].isin(test_list)]

        train_dataset = self._final_prepare_dataset(train_dataset)
        test_dataset = self._final_prepare_dataset(test_dataset)

        return test_dataset, train_dataset

    def _final_prepare_dataset(self, dataset):
        '''
            Clean Up Useless columns and final prep the dataset.
        '''
        assert 'full_path' in dataset.columns and 'Finding Labels' in dataset.columns, "Key Columns are missing."
        f_dataset = dataset[['full_path','Finding Labels']]
        f_dataset = datasets.Dataset.from_pandas(f_dataset)

        return f_dataset


    def _generate_dataset(self):
        '''
            Generate datasets through Kaggle, Data size: about 50 GB.
        '''
        target = 'Data_Entry_2017.csv'
        for root, dirs, files in os.walk(self.path):
            if target in dirs or target in files:
                return


        api = KaggleApi()
        api.authenticate()

        os.makedirs(self.path, exist_ok=True)

        api.dataset_download_files(
            "nih-chest-xrays/data", path=self.path, unzip=True
        )


# if __name__ == '__main__':
#     from multimedeval import MultiMedEval, SetupParams, EvalParams
#     engine = MultiMedEval()
#
#     setupParams = SetupParams(chestray14_dir="../data/chestxray14/")
#     tasksReady = engine.setup(setup_params= setupParams)
#
#
#     def batcher(prompts) -> list[str]:
#         return ["Answer" for _ in prompts]
#
#
#     evalParams = EvalParams(batch_size=128)
#     results = engine.eval(["ChestXray14"], batcher, eval_params=evalParams)
#     results
"""The QA task family."""

from datasets import Dataset, load_dataset
from torchmetrics.text import BLEUScore

from multimedeval.task_families import QA
from multimedeval.utils import clean_str, BatcherInput


class MedQA(QA):
    """The MedQA task."""

    def __init__(self, **kwargs):
        """Initialize the MedQA task."""
        super().__init__(**kwargs)
        self.task_name = "MedQA"
        self.modality = "General medicine"
        self.bleu_scorer = BLEUScore(n_gram=1)

    def setup(self):
        """Setup the MedQA task family."""
        cache_dir = self.engine.get_config()["medqa_dir"]

        if cache_dir is None:
            raise ValueError(
                "No path for MedQA dataset provided in the config file. Skipping the task."
            )

        self.dataset = load_dataset(
            "bigbio/med_qa",
            name="med_qa_en_source",
            split="test",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        self.train_dataset = load_dataset(
            "bigbio/med_qa",
            name="med_qa_en_source",
            split="train",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

    def format_question(self, sample, prompt=False):
        """Format the question for the MedQA task.

        Args:
            sample: The sample to format.
            prompt: Whether or not to add the answer in the prompt. Defaults to False.

        Returns:
            An instance of BatcherInput with The formatted question.
        """
        question = sample["question"]
        options = sample["options"]

        formatted_question = f"{question}\n"
        formatted_question += (
            "Options:\n"
            + "\n".join([f'{option["key"]}: {option["value"]}.' for option in options])
            + "\n"
        )
        formatted_question += "What is the correct answer?"
        batcher_input = BatcherInput()

        batcher_input._add_text_prompt('user',formatted_question)

        # question = [{"role": "user", "content": formatted_question}]
        if prompt:
            formatted_answer = "The answer is " + sample["answer_idx"] + "."
            # question.append({"role": "assistant", "content": formatted_answer})
            batcher_input._add_text_prompt('assistant', formatted_answer)
        return batcher_input

    def get_correct_answer(self, sample, full_text=False):
        """Get the correct answer for the sample.

        Args:
            sample: The sample to get the correct answer for.
            fullText: Whether or not to return the raw text answer. Defaults to False.

        Returns:
            The correct answer.
        """
        if full_text:
            return f"{sample['answer_idx']}: {sample['answer'].lower().strip()}"

        return sample["answer_idx"].lower().strip()

    def get_predicted_answer(self, pred: str, sample):
        """Get the answer predicted by the model.

        Args:
            pred: The generated answer.
            sample: The sample used to generate the answer.

        Returns:
            The predicted answer.
        """
        pred = clean_str(pred)
        if len(pred) == 0:
            return "Invalid answer"

        options = [
            clean_str(f'{option["key"]} {option["value"]}')
            for option in sample["options"]
        ]
        # Compute the BLEU score for each option
        scores = [self.bleu_scorer([pred], [[option]]) for option in options]

        if max(scores) == 0:
            return "Invalid answer"

        pred = sample["options"][scores.index(max(scores))]["key"].lower()

        return pred


class PubMedQA(QA):
    """The PubMedQA task."""

    def __init__(self, **kwargs):
        """Initialize the PubMedQA task."""
        super().__init__(**kwargs)
        self.task_name = "PubMedQA"
        self.modality = "General medicine"
        self.task = "QA"

    def setup(self):
        """Setup the PubMedQA task."""
        cache_dir = self.engine.get_config()["pubmedqa_dir"]

        if cache_dir is None:
            raise ValueError(
                "No path for MedQA dataset provided in the config file. Skipping the task."
            )

        self.dataset = load_dataset(
            "bigbio/pubmed_qa",
            name="pubmed_qa_labeled_fold1_bigbio_qa",
            split="test",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        self.train_dataset = load_dataset(
            "bigbio/pubmed_qa",
            name="pubmed_qa_labeled_fold1_bigbio_qa",
            split="train",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

    def get_correct_answer(self, sample, full_text=False):
        """Get the correct answer for the sample.

        Args:
            sample: The sample to get the correct answer for.
            fullText: Whether or not to return the raw text answer. Defaults to False.

        Returns:
            The correct answer.
        """
        return sample["answer"][0]

    def format_question(self, sample, prompt=False):
        """Format the question for the PubMedQA task.

        Args:
            sample: The sample to format.
            prompt: Whether or not to add the answer in the prompt. Defaults to False.

        Returns:
            An instance of BatcherInput with the formatted question.
        """
        context = sample["context"]
        question = sample["question"]
        answer = sample["answer"]

        formatted_question = "Answer the question with yes, no or maybe. "
        formatted_question += f"{context} {question}"

        formatted_answer = answer[0]

        # question = [{"role": "user", "content": formatted_question}]
        batcher_input = BatcherInput()

        batcher_input._add_text_prompt('user',formatted_question)
        if prompt:
            # question.append({"role": "assistant", "content": formatted_answer})
            batcher_input._add_text_prompt('assistant', formatted_answer)
        return batcher_input

    def get_predicted_answer(self, pred: str, sample):
        """Get the answer predicted by the model.

        Args:
            pred: The generated answer.
            sample: The sample used to generate the answer.

        Returns:
            The predicted answer.
        """
        pred = clean_str(pred)
        if len(pred) == 0:
            return "Invalid answer"

        option_vocabs = [["yes"], ["no"], ["maybe"]]

        pred_tokens = pred.split(" ")
        scores = [0 for _ in range(len(option_vocabs))]
        for token in pred_tokens:
            for idx, option_vocab in enumerate(option_vocabs):
                if token in option_vocab:
                    scores[idx] += 1

        if max(scores) == 0:
            return "Invalid answer"

        pred = option_vocabs[scores.index(max(scores))][0].lower()
        return pred


class MedMCQA(QA):
    """The MedMCQA task."""

    def __init__(self, **kwargs):
        """Initialize the MedMCQA task."""
        super().__init__(**kwargs)
        self.task_name = "MedMCQA"
        self.modality = "General medicine"
        self.task = "QA"
        self.bleu_scorer = BLEUScore(n_gram=1)

    def setup(self):
        """Setup the MedMCQA task."""
        cache_dir = self.engine.get_config()["medmcqa_dir"]

        if cache_dir is None:
            raise ValueError(
                "No path for MedQA dataset provided in the config file. Skipping the task."
            )

        self.dataset = load_dataset("medmcqa", split="validation", cache_dir=cache_dir)

        self.train_dataset = load_dataset("medmcqa", split="train", cache_dir=cache_dir)

    def format_question(self, sample, prompt=False):
        """Format the question for the MedMCQA task.

        Args:
            sample: The sample to format.
            prompt: Whether or not to add the answer in the prompt. Defaults to False.

        Returns:
            An instance of BatcherInput with formatted question.
        """
        question = sample["question"]
        options = self._get_options(sample)
        answer = sample["cop"]

        formatted_question = f"{question}\n"
        formatted_question += "\n".join(options) + "\n"
        formatted_question += "What is the correct answer?"

        formatted_answer = f"The answer is {options[answer]}."

        # question = [{"role": "user", "content": formatted_question}]

        batcher_input = BatcherInput()

        batcher_input._add_text_prompt('user',formatted_question)
        if prompt:
            # question.append({"role": "assistant", "content": formatted_answer})
            batcher_input._add_text_prompt('assistant', formatted_answer)
        return batcher_input

    def get_correct_answer(self, sample, full_text=False):
        """Get the correct answer for the sample.

        Args:
            sample: The sample to get the correct answer for.
            fullText: Whether or not to return the raw text answer. Defaults to False.

        Returns:
            The correct answer.
        """
        number = sample["cop"]
        if full_text:
            return self._get_options(sample)[number]

        return str(number + 1)

    def _get_options(self, sample):
        return [
            f"a: {sample['opa']}.",
            f"b: {sample['opb']}.",
            f"c: {sample['opc']}.",
            f"d: {sample['opd']}.",
        ]

    def get_predicted_answer(self, pred: str, sample):
        """Get the answer predicted by the model.

        Args:
            pred: The generated answer.
            sample: The sample used to generate the answer.

        Returns:
            The predicted answer.
        """
        pred = clean_str(pred)
        if len(pred) == 0:
            return "Invalid answer"

        # Compute the BLEU score for each option
        scores = [
            self.bleu_scorer([pred], [[clean_str(option)]])
            for option in self._get_options(sample)
        ]

        if max(scores) == 0:
            return "Invalid answer"

        pred = str(
            scores.index(max(scores)) + 1
        )  # +1 because the options are 1, 2, 3, 4 and not 0, 1, 2, 3
        return pred


class MMLU(QA):
    """The MMLU task."""

    def __init__(self, **kwargs):
        """Initialize the MMLU task."""
        super().__init__(**kwargs)
        self.task_name = "MMLU"
        self.modality = "General knowledge"
        self.task = "QA"
        self.bleu_scorer = BLEUScore(n_gram=1)

    def setup(self):
        """Setup the MMLU task."""
        cache_dir = self.engine.get_config()["mmlu_dir"]

        if cache_dir is None:
            raise ValueError(
                "No path for MMLU dataset provided in the config file. Skipping the task."
            )

        self.dataset = load_dataset(
            "cais/mmlu", "all", split="test", cache_dir=cache_dir
        )
        self.dataset = self.dataset.to_pandas()

        def keep_quarter(group):
            quarter_len = len(group) // 4  # Calculate the length of a quarter
            return group.head(quarter_len)  # Keep the first quarter of rows

        self.dataset = self.dataset.groupby("subject").apply(keep_quarter)
        self.dataset = Dataset.from_pandas(self.dataset)

        self.train_dataset = load_dataset(
            "cais/mmlu", "all", split="dev", cache_dir=cache_dir
        )

    def format_question(self, sample, prompt=False):
        """Format the question for the MMLU task.

        Args:
            sample: The sample to format.
            prompt: Whether or not to add the answer in the prompt. Defaults to False.

        Returns:
            An instance of BatcherInput with formatted question.
        """
        question = sample["question"]
        options = self._get_options(sample)
        answer = sample["answer"]  # Answer is the index of the correct option

        formatted_question = f"{question}\n"
        formatted_question += "\n".join(options) + "\n"
        formatted_question += "Answer:"

        formatted_answer = f"The answer is {options[answer]}."

        # question = [{"role": "user", "content": formatted_question}]
        batcher_input = BatcherInput()

        batcher_input._add_text_prompt('user', formatted_question)
        if prompt:
            # question.append({"role": "assistant", "content": formatted_answer})
            batcher_input._add_text_prompt('assistant', formatted_answer)
        return batcher_input

    def get_correct_answer(self, sample, full_text=False):
        """Get the correct answer for the sample.

        Args:
            sample: The sample to get the correct answer for.
            fullText: Whether or not to return the raw text answer. Defaults to False.

        Returns:
            The correct answer.
        """
        number = sample["answer"]
        if full_text:
            return self._get_options(sample)[number]
        return str(number + 1)

    def _get_options(self, sample):
        choices = sample["choices"]
        letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        options = [f"{idx}. {choice}." for idx, choice in zip(letters, choices)]
        return options

    def get_predicted_answer(self, pred: str, sample):
        """Get the answer predicted by the model.

        Args:
            pred: The generated answer.
            sample: The sample used to generate the answer.

        Returns:
            The predicted answer.
        """
        pred = clean_str(pred)
        if len(pred) == 0:
            return "Invalid answer"

        # Compute the BLEU score for each option
        scores = [
            self.bleu_scorer([pred], [[clean_str(option)]])
            for option in self._get_options(sample)
        ]

        if max(scores) == 0:
            return "Invalid answer"

        pred = str(scores.index(max(scores)) + 1)

        return pred

# MultiMedEval


MultiMedEval is a library to evaluate the performance of Vision-Language Models (VLM) on medical domain tasks. The goal is to have a set of benchmark with a unified evaluation scheme to facilitate the development and comparison of medical VLM.
We include 12 tasks representing a range of different imaging modalities.


## Tasks

| Task                           | Description                                                                                                        | Modality       | Size
|--------------------------------|--------------------------------------------------------------------------------------------------------------------|----------------|----------
| MedQA                          | Multiple choice questions on general medical knowledge                                                             | Text           |
| PubMedQA                       | Yes/no/maybe questions based on PubMed paper abstracts                                                             | Text           |
| MedMCQA                        | Multiple choice questions on general medical knowledge                                                             | Text           |
| MIMIC-CXR-ReportGeneration     | Generation of finding sections of radiology reports based on the radiology images                                  | Chest X-ray    |
| VQA-RAD                        | Open ended questions on radiology images                                                                           | X-ray          |
| Path-VQA                       | Open ended questions on pathology images                                                                           | Pathology      |
| SLAKE                          | Open ended questions on radiology images                                                                           | X-ray          |
| MIMIC-CXR-ImageClassification  | Classification of radiology images into 5 diseases                                                                 | Chest X-ray    |
| VinDr-Mammo                    | Classification of mammography images into 5 BIRADS levels                                                          | Mammography    |
| Pad-UFES-20                    | Classification of skin lesion images into 7 diseases                                                               | Dermatology    |
| CBIS-DDSM-Mass                 | Classification of masses in mammography images into "benign", "malignant" or "benign without callback"             | Mammography    |
| CBIS-DDSM-Calcification        | Classification of calcification in mammography images into "benign", "malignant" or "benign without callback"      | Mammography    |
| MIMIC-III                      | Summarization of radiology reports                                                                                 | Text           |
| MedNLI                         | Natural Language Inference on medical sentences.                                                                   | Text           |
| MNIST-Oct                      |                                                                                                                    | OCT            |
| MNIST-Path                     |                                                                                                                    | Pathology      |
| MNIST-Blood                    |                                                                                                                    | Microscopy     |
| MNIST-Breast                   |                                                                                                                    | Mammography    |
| MNIST-Derma                    |                                                                                                                    | Dermatology    |
| MNIST-OrganC                   |                                                                                                                    | CT             |
| MNIST-OrganS                   |                                                                                                                    | CT             |
| MNIST-Pneumonia                |                                                                                                                    | X-Ray          |
| MNIST-Retina                   |                                                                                                                    | Fondus Camera  |
| MNIST-Tissue                   |                                                                                                                    | Microscopy     |



<p align="center">
    <img src="figures/sankey.png" alt="sankey graph">
    <br>
    <em>Representation of the modalities, tasks and datasets in MultiMedEval</em>
</p>



## Setup

To install the library, you can use `pip`

```console
pip install git+https://github.com/corentin-ryr/MultiMedEval.git
```

The setup script needs a configuration file containing the destination folder for every dataset. You need to specify this config file manually to fit your system. The config file follows this example:
```json
{
  "huggingfaceCacheDir": {"path": ""},
  "physionet": {"path": "", "username": "", "password": ""},
  "SLAKE": {"path": ""},
  "MIMIC-CXR": {"path":"/PATH/TO/DOCUMENTS/"}, // Optional (will download to physionet folder if the line is not present): example if /PATH/TO/DOCUMENTS/mimic-cxr-jpg/2.0.0
  "SLAKE": {"path": ""},
  "CheXBert": {"dlLocation": ""},
  "Pad-UFES-20": {"path": ""},
  "CBIS-DDSM": {"path": ""},
  "MedMNISTCacheDir": {"path": ""},
  "tasksToPrepare": []
}
```
`TasksToPrepare` defines the list of the tasks that will be downloaded and available for evaluation. If the list if empty, all tasks will be prepared.

To download the datasets and prepare the evaluation models, you can instantiate the main "engine" without any parameters. This will run the setup for all tasks but not the evaluation itself.

```python
from multimedeval import MultiMedEval, Params

engine = MultiMedEval()
```

During the setup process, the script will need a Physionet username and password to download "VinDr-Mammo", "MIMIC-CXR" and "MIMIC-III".
You also need to setup Kaggle on your machine before running the setup as the "CBIS-DDSM" is hosted on Kaggle.

At the end of the setup process, you will see a summary of which tasks are ready and which didn't run properly.

## Usage

The user must implement one function: `batcher`. It takes a batch of input and must return the answer.
The batch is a list of inputs.
Each input is a tuple of:
* a prompt in the form of a conversation between a user and an assistant.
* a list of Pillow images. The number of images matches the number of <img> tokens in the prompt.

```python
[
    (
        [
            {"role": "user", "content": "This is a question with an image <img>."}, 
            {"role": "assistant", "content": "This is the answer."},
            {"role": "user", "content": "This is a question with an image <img>."}, 
        ], 
        [PIL.Image(), PIL.Image()]
    ),
    (
        [
            {"role": "user", "content": "This is a question without images."},
            {"role": "assistant", "content": "This is the answer."},
            {"role": "user", "content": "This is a question without images."}, 
        ], 
        []
    ),

]
```

Here is an example of a `batcher` without any logic:
```python
def batcher(prompts:list[tuple]) -> list[str]:
    return ["Answer" for _ in prompts]
``` 

Here is anotther example of a `batcher` this time implemented as a callable class. It initializes the Mistral model (a language-only model) and queries it in the `__call__` function.

```python
class batcherMistral:
    def __init__(self) -> None:
        self.model: MistralModel = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, prompts):
        model_inputs = [self.tokenizer.apply_chat_template(messages[0], return_tensors="pt", tokenize=False) for messages in prompts]
        model_inputs = self.tokenizer(model_inputs, padding="max_length", truncation=True, max_length=1024, return_tensors="pt")

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=200, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        # Remove the first 1024 tokens (prompt)
        generated_ids = generated_ids[:, model_inputs["input_ids"].shape[1] :]

        answers = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return answers
```

To run the benchmark, call the `eval` method of the `MultiMedEval` class with the list of tasks to benchmark on. If the list is empty, all the tasks will be benchmarked.

```python
from multimedeval import MultiMedEval, Params
engine = MultiMedEval(params=Params(), batcher=batcherMistral())

engine.eval(["MedQA", "VQA-RAD", "MIMIC-CXR-ReportGeneration"])
```

## MultiMedEval parameters

The `Params` class takes the following arguments:
* batch_size: an int initialized as `128` and representing the number of prompt sent to the batcher at once.
* run_name: a string initialized as the current date. It will be the name of the folder where the results are saved.
* fewshot: a bool to indicate whether or not to add example prompt before test prompts.
* num_workers: an int defining how many workers to use in the dataloader.
* device: a string defining where to run the evaluation.

## References


## Related work

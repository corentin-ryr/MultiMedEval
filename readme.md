# MultiMedBench

MultiMedBench is a benchmark for medical conversational models.


## Tasks

### Question Answering

These tasks are text only.

| Dataset  | Modality | Size     |
| -------- | -------- | -------- |
| MedQA    | Text     | 1,273    |
| MedMCQA  | Text     | 4,183    |
| PubMedQA | Text     | 500      |

Benchmark:
| Model    | Mistral7B | MedAlpaca     |
| -------- | --------- | ------------- |
| MedQA    | 0.359     |               |
| MedMCQA  |           |               |
| PubMedQA | 0.398     |               |


### Report Summarization

| Dataset  | Modality | Size     |
| -------- | -------- | -------- |
| MIMIC-III | CT & MRI |  13,057 |


### Visual Question Answering

| Dataset  | Modality | Size     |
| -------- | -------- | -------- |
| VQA-RAD  |  | 451 |
| Slake-VQA | CT, MRI, and chest X-rays |  2,070 |
| Path-VQA |  | 6,761 |


### Report Generation

| Dataset  | Modality | Size     |
| -------- | -------- | -------- |
| MIMIC-CXR | Chest X-ray | 4,834 |


### Medical Image Classification

| Dataset  | Modality | Size     |
| -------- | -------- | -------- |
| MIMIC-CXR    |  | Official |
| PAD-UFES-20  | Dermatology | 460 |
| VinDr-Mammo |  | 4,000 |
| CBIS-DDSM (mass) |  | 378 |
| CBIS-DDSM (calcification) |  | 326 |
| PrecisionFDA Truth Challenge V2 |  | 13,030 |


## Usage

The user must implement two functions: `batcher` and `prepare`.

### Batcher

The `batcher` function takes a batch of input and must return the answer.
The batch is a list of inputs.
The input is a tuple of:
* a prompt in the form of a text that may or may not refer to images.
* a dictionary of id: images
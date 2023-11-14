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
| Model    | Mistral7B | MedAlpaca     | Llama 2       |
| -------- | --------- | ------------- | ------------- |
| MedQA    | 0.361     | 0.239         | 0.296         |
| PubMedQA | 0.396     | 0.474         | 0.488         |
| MedMCQA  |           | 0.301         |          |


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



## Inference speed


For Llama2 on a V100 with 32G it takes 18min to benchmark MedQA:
* batch size: 36
* 125 token/second
* There is some problems when using float16 (sometimes it gives nan values in logits). The solution is to use bfloat16 (it was trained using that) but the format is poorly supported on v100 and earlier GPUs (it reduces the performance by 50%).
* Training on A100 XXX


## Indexing problem

There use to be an indexing problem because we need to check that the generation_config is good.
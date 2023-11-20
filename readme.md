# MultiMedBench

MultiMedBench is a benchmark for medical conversational models.


## Tasks

![metrics](figures/metrics.png)

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
| MedQA    | 0.344     | 0.248         | 0.305         |
| PubMedQA | 0.708     | 0.628         | 0.652         |
| MedMCQA  | 0.417     | 0.293         | 0.338         |


### Report Summarization

| Dataset  | Modality | Size     |
| -------- | -------- | -------- |
| MIMIC-III | CT & MRI |  13,057 |


### Visual Question Answering

| Dataset   | Modality                  | Size     |
| --------- | ------------------------- | -------- |
| VQA-RAD   | CT, MRI, and chest X-rays | 451      |
| Slake-VQA | CT, MRI, and chest X-rays | 2,070    |
| Path-VQA  | Pathology                 | 6,761    |


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

The user must implement one functions: `batcher`.

### Batcher

The `batcher` function takes a batch of input and must return the answer.
The batch is a list of inputs.
The input is a tuple of:
* a prompt in the form of a text that may or may not refer to images.
* a dictionary of id: images



## Inference speed


For Llama2 on a V100 with 32GB it takes 18min to benchmark MedQA:
* batch size: 36 (with context length 512)
* 125 token/second (with float 16 but 60t/s with bfloat)
* There is some problems when using float16 (sometimes it gives nan values in logits). The solution is to use bfloat16 (it was trained using that) but the format is poorly supported on v100 and earlier GPUs (it reduces the performance by 50%).

For Llama2 on a A100 with 80GB it takes min to benchmark MedQA:
* batch size of 50 (with context length 1024)
* 200t/s
* bfloat16 is supported



## Indexing problem

There use to be an indexing problem because we need to check that the generation_config is good.
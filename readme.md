# MultiMedBench

MultiMedBench is a benchmark for medical conversational models.


## Usage

The user must implement one functions: `batcher`. It takes a batch of input and must return the answer.
The batch is a list of inputs.
The input is a tuple of:
* a prompt in the form of a text that may or may not refer to images.
* a dictionary of id: images

```python
[
    (
        [
            {"role": "user", "content": "This is a question with an image <img>."}, 
            {"role": "asssitant", "content": "This is the answer."},
            {"role": "user", "content": "This is a question with an image <img>."}, 
        ], 
        [PIL.Image(), PIL.Image()]
    ),
    (
        [
            {"role": "user", "content": "This is a question without images."},
            {"role": "asssitant", "content": "This is the answer."},
            {"role": "user", "content": "This is a question without images."}, 
        ], 
        []
    ),

]


```



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

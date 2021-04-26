# presumm_bertext_inference
The code is for the inference pipeline of [PreSumm](https://github.com/nlpyang/PreSumm/tree/dev) Bert extractive summarization

**Note**: For inference of BertSumExt, max_pos is at most 512 tokens.

Codes are borrowed from [PreSumm](https://github.com/nlpyang/PreSumm/tree/dev)



## Model Checkpoints

[CNN/DM Extractive](https://drive.google.com/open?id=1kKWoV0QCbeIuFt85beQgJ4v0lujaXobJ)

Move the model checkpoints to `models/bertext_cnndm_transformer.pt`



## Format

* Input: text file, each line is the text of a document.

* Output: text file, each line is the output summary of the document (sentences are split by '\<q\>').



## Run

### Command line

```
python3 call.py -input_file_path ${Input_file_path} -output_file_path ${Output_file_path}
```

### Python3 interface

```
from call import run_summarize
run_summarize(input_file_path, output_file_path)
```
import argparse
import os
import subprocess
from shutil import copyfile
from spacy.lang.en import English
from typing import List

class SentenceSplitter:
    def __init__(self, language=English):
        self.nlp = language()
        try:
            self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
            self.is_spacy_3 = False
        except:
            self.nlp.add_pipe("sentencizer")
            self.is_spacy_3 = True
    
    def split(self, body: str):
        sentences = []
        for c in self.nlp(body).sents:
            if self.is_spacy_3:
                sentences.append(c.text.strip())
            else:
                sentences.append(c.string.strip())
        return sentences

    def __call__(self, body: str) -> List[str]:
        return self.split(body)

def combine_sentences(sentences, token=" [CLS] [SEP] "):
    return token.join(sentences)

def preprocess_input(input_file_path, processed_input_file_path, split_fn):
    f_input = open(input_file_path, "r", encoding="utf-8")
    f_output = open(processed_input_file_path, "w", encoding="utf-8")
    for doc in f_input:
        doc = doc.strip()
        doc_sents = split_fn(doc)
        processed_doc_str = combine_sentences(doc_sents)
        f_output.write(processed_doc_str + "\n")
    f_input.close()
    f_output.close()

def clean_directory(path, exclusion=[]):
    exclusion.append(".gitignore")
    for filename in os.listdir(path):
        if filename not in exclusion:
            os.remove(os.path.join(path, filename))

def run_summarize(input_file_path, output_file_path):
    # workingpath = os.getcwd()
    curpath = os.path.dirname(os.path.realpath(__file__))

    # split sentence handler
    sentsplitter = SentenceSplitter()

    # preprocess input for BertExt
    processed_input_file_path = input_file_path + ".processed"
    preprocess_input(input_file_path, processed_input_file_path, split_fn=sentsplitter)

    # run BertExt summarizer
    result_path = os.path.join(curpath, "results/")
    # run command with absolute path
    # python3 src/train.py -task ext -mode test_text -test_from models/bertext_cnndm_transformer.pt -text_src $SRC -result_path results/ -max_pos 512
    command = ['python3', os.path.join(curpath, "src/train.py"),
               '-task', 'ext',
               '-mode', 'test_text',
               '-test_from', os.path.join(curpath, "models/bertext_cnndm_transformer.pt"),
               '-text_src', processed_input_file_path,
               '-result_path', result_path,
               '-max_pos', '512']
    subprocess.call(command)

    # copy output summary file
    for filename in os.listdir(result_path):
        if filename.endswith(".candidate"):
            copyfile(os.path.join(result_path, filename), output_file_path)
    
    # clean directories
    if os.path.exists(processed_input_file_path):
        os.remove(processed_input_file_path)
    clean_directory(os.path.join(curpath, "models"), exclusion=["bertext_cnndm_transformer.pt"])
    clean_directory(os.path.join(curpath, "logs"))
    clean_directory(result_path)
    print(curpath)
    #clean_directory(os.path.join(curpath, "temp"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_file_path", required=True, type=str, help='Input document path.')
    parser.add_argument("-output_file_path", required=True, type=str, help='Path to save summaries.')
    args = parser.parse_args()

    run_summarize(args.input_file_path, args.output_file_path)
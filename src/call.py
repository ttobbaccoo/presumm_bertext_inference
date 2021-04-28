import argparse
import os
import subprocess
import torch
from io import StringIO
from shutil import copyfile
from spacy.lang.en import English
from tqdm import tqdm
from typing import List

from models.data_loader import Batch
from models.model_builder import ExtSummarizer
from models.trainer_ext import build_trainer
from others.logging import logger, init_logger
from train import str2bool

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

def get_preprocessed_input(input_file_path, split_fn):
    f_input = open(input_file_path, "r", encoding="utf-8")
    processed_input_str = []
    for doc in f_input:
        doc = doc.strip()
        doc_sents = split_fn(doc)
        processed_doc_str = combine_sentences(doc_sents)
        processed_input_str.append(processed_doc_str)
    f_input.close()
    processed_input_file = StringIO("\n".join(processed_input_str))
    return processed_input_file

def clean_directory(path, exclusion=[]):
    exclusion.append(".gitignore")
    for filename in os.listdir(path):
        if filename not in exclusion:
            os.remove(os.path.join(path, filename))

def run_summarize_command(input_file_path, output_file_path):
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

def parse_arguments(dirpath, visible_gpus):
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", default='ext', type=str, choices=['ext', 'abs'])
    parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
    parser.add_argument("-mode", default='test_text', type=str, choices=['train', 'validate', 'test', 'test_text'])
    parser.add_argument("-bert_data_path", default=os.path.join(dirpath, 'bert_data_new/cnndm'))
    parser.add_argument("-model_path", default=os.path.join(dirpath, 'models/'))
    parser.add_argument("-result_path", default=os.path.join(dirpath, 'results/'))
    parser.add_argument("-temp_dir", default=os.path.join(dirpath, 'temp'))
    parser.add_argument("-text_src", default='')
    parser.add_argument("-text_tgt", default='')

    parser.add_argument("-batch_size", default=140, type=int)
    parser.add_argument("-test_batch_size", default=200, type=int)
    parser.add_argument("-max_ndocs_in_batch", default=6, type=int)

    parser.add_argument("-max_pos", default=512, type=int)
    parser.add_argument("-use_interval", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-large", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-load_from_extractive", default='', type=str)

    parser.add_argument("-sep_optim", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-lr_bert", default=2e-3, type=float)
    parser.add_argument("-lr_dec", default=2e-3, type=float)
    parser.add_argument("-use_bert_emb", type=str2bool, nargs='?',const=True,default=False)

    parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-dec_dropout", default=0.2, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=8, type=int)
    parser.add_argument("-dec_ff_size", default=2048, type=int)
    parser.add_argument("-enc_hidden_size", default=512, type=int)
    parser.add_argument("-enc_ff_size", default=512, type=int)
    parser.add_argument("-enc_dropout", default=0.2, type=float)
    parser.add_argument("-enc_layers", default=6, type=int)

    # params for EXT
    parser.add_argument("-ext_dropout", default=0.2, type=float)
    parser.add_argument("-ext_layers", default=2, type=int)
    parser.add_argument("-ext_hidden_size", default=768, type=int)
    parser.add_argument("-ext_heads", default=8, type=int)
    parser.add_argument("-ext_ff_size", default=2048, type=int)

    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha",  default=0.6, type=float)
    parser.add_argument("-beam_size", default=5, type=int)
    parser.add_argument("-min_length", default=15, type=int)
    parser.add_argument("-max_length", default=150, type=int)
    parser.add_argument("-max_tgt_len", default=140, type=int)



    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=1, type=float)
    parser.add_argument("-beta1", default= 0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-warmup_steps_bert", default=8000, type=int)
    parser.add_argument("-warmup_steps_dec", default=8000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-save_checkpoint_steps", default=5, type=int)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-report_every", default=1, type=int)
    parser.add_argument("-train_steps", default=1000, type=int)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)


    parser.add_argument('-visible_gpus', default=visible_gpus, type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_file', default=os.path.join(dirpath, 'logs/cnndm.log'))
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-test_from", default=os.path.join(dirpath, "models/bertext_cnndm_transformer.pt"))
    parser.add_argument("-test_start_from", default=-1, type=int)

    parser.add_argument("-train_from", default='')
    parser.add_argument("-report_rouge", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    init_logger(args.log_file)
    return args

def load_text(args, source_fp, device):
    from others.tokenization import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    sep_vid = tokenizer.vocab['[SEP]']
    cls_vid = tokenizer.vocab['[CLS]']
    n_lines = len(source_fp.getvalue().split('\n'))

    def _process_src(raw):
        raw = raw.strip().lower()
        raw = raw.replace('[cls]','[CLS]').replace('[sep]','[SEP]')
        src_subtokens = tokenizer.tokenize(raw)
        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']
        src_subtoken_idxs = tokenizer.convert_tokens_to_ids(src_subtokens)
        src_subtoken_idxs = src_subtoken_idxs[:-1][:args.max_pos]
        src_subtoken_idxs[-1] = sep_vid
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        segs = segs[:args.max_pos]
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        src = torch.tensor(src_subtoken_idxs)[None, :].to(device)
        mask_src = (1 - (src == 0).float()).to(device)
        cls_ids = [[i for i, t in enumerate(src_subtoken_idxs) if t == cls_vid]]
        clss = torch.tensor(cls_ids).to(device)
        mask_cls = 1 - (clss == -1).float()
        clss[clss == -1] = 0

        return src, mask_src, segments_ids, clss, mask_cls

    for x in tqdm(source_fp, total=n_lines):
        src, mask_src, segments_ids, clss, mask_cls = _process_src(x)
        segs = torch.tensor(segments_ids)[None, :].to(device)
        batch = Batch()
        batch.src  = src
        batch.tgt  = None
        batch.mask_src  = mask_src
        batch.mask_tgt  = None
        batch.segs  = segs
        batch.src_str  =  [[sent.replace('[SEP]','').strip() for sent in x.split('[CLS]')]]
        batch.tgt_str  = ['']
        batch.clss  = clss
        batch.mask_cls  = mask_cls

        batch.batch_size=1
        yield batch

def run_summarize(input_file_path, output_file_path, visible_gpus='-1'):
    dirpath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    # split sentence handler
    sentsplitter = SentenceSplitter()

    # preprocess input for BertExt
    processed_input_file_path = get_preprocessed_input(input_file_path, split_fn=sentsplitter)

    args = parse_arguments(dirpath, visible_gpus)

    logger.info('Loading checkpoint from %s' % args.test_from)
    checkpoint = torch.load(args.test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval', 'rnn_size']
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    model = ExtSummarizer(args, device, checkpoint)
    model.eval()

    test_iter = load_text(args, processed_input_file_path, device)

    trainer = build_trainer(args, device_id, model, None)
    trainer.test(test_iter, -1)

    # postprocess
    # copy output summary file
    for filename in os.listdir(args.result_path):
        if filename.endswith(".candidate"):
            copyfile(os.path.join(args.result_path, filename), output_file_path)
    
    # clean directories
    clean_directory(os.path.join(dirpath, "models"), exclusion=["bertext_cnndm_transformer.pt"])
    clean_directory(os.path.join(dirpath, "logs"))
    clean_directory(args.result_path)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-input_file_path", required=True, type=str, help='Input document path.')
    # parser.add_argument("-output_file_path", required=True, type=str, help='Path to save summaries.')
    # args = parser.parse_args()
    # run_summarize_command(args.input_file_path, args.output_file_path)

    run_summarize("data1.txt", "data2.txt")
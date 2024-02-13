import argparse
from glob import glob
import gzip
import logging
import os

import pandas as pd
import torch
import tqdm
from transformers import (
    T5Tokenizer,
    MT5Tokenizer,
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
)


logging.basicConfig(level=logging.INFO)
LANGUAGES = (
    'german',
    'norwegian1',
    'norwegian2',
    'russian',
    'english',
    'latin',
    'italian',
    'swedish',
)

MODELS = {
    "english": "mt0-definition-en-xl",
    #"english": "flan-t5-definition-en-base",
    "norwegian1": "mt0-definition-no-xl",
    "norwegian2": "mt0-definition-no-xl",
    "russian": "mt0-definition-ru-xl",
    "russian2": "mt0-definition-ru-xl",
    "russian3": "mt0-definition-ru-xl",
}


def load_model_and_tokenizer(model_path):
    tokenizer = T5Tokenizer.from_pretrained(
        model_path,
        add_prefix_space=True,
    )
    if torch.cuda.is_available():
        model = T5ForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
        )
    else:
        model = T5ForConditionalGeneration.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
        )
    model.eval()
    return model, tokenizer


def load_mt5_model_and_tokenizer(model_path):
    tokenizer = MT5Tokenizer.from_pretrained(
        model_path,
        add_prefix_space=True,
    )
    if torch.cuda.is_available():
        logging.info("Predicting on cuda")
        model = MT5ForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
        )
    else:
        model = MT5ForConditionalGeneration.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
        )
    model.eval()
    return model, tokenizer


def parse_arge():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang",
        default="english",
    )
    parser.add_argument(
        "--data_path",
        default="../acl_data",
    )
    parser.add_argument(
        "--bsize",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--maxl",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--res_path",
        default="./"
    )
    parser.add_argument(
        "--n_first",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--models_dir",
        default="ltg",
    )
    parser.add_argument(
        "--max_new_tokens",
        default=50,
        type=int,
    )
    parser.add_argument(
        "--decoding_strategy",
        default="greedy",
        choices=("greedy", "beam", "diverse_beam"),
    )
    parser.add_argument(
        "--num_beams",
        default=1,  # no beam search by default
        type=int,
    )
    parser.add_argument(
        "--num_beam_groups",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--diversity_penalty",
        default=0.0,
        type=float,
    )
    return parser.parse_args()


def define(
        in_prompts,
        lm,
        cur_tokenizer,
        arguments,
        targets,
        filter_target=True,
):
    logging.info(f"Tokenizing with max length {arguments.maxl}...")
    inputs = cur_tokenizer(
        in_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=arguments.maxl,
    )
    logging.info("Tokenizing finished.")

    target_ids = cur_tokenizer(targets, add_special_tokens=False).input_ids
    target_ids = torch.tensor([el[-1] for el in target_ids])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    test_dataset = torch.utils.data.TensorDataset(
        inputs["input_ids"].to(device),
        inputs["attention_mask"].to(device),
        target_ids.to(device),
    )
    test_iter = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=arguments.bsize,
                                            shuffle=False)
    logging.info(f"Generating definitions with batch size {arguments.bsize}...")

    definitions = []
    for inp, att, targetwords in tqdm.tqdm(test_iter):
        with torch.no_grad():
            if filter_target:
                bad = [[el] for el in targetwords.tolist()]
                outputs = lm.generate(
                    input_ids=inp,
                    attention_mask=att,
                    do_sample=False,
                    bad_words_ids=bad,
                    max_new_tokens=arguments.max_new_tokens,
                    num_beams=arguments.num_beams,
                    num_beam_groups=arguments.num_beam_groups,
                    diversity_penalty=arguments.diversity_penalty,
                )
            else:
                outputs = lm.generate(input_ids=inp, attention_mask=att,
                                      do_sample=False)
        predictions = cur_tokenizer.batch_decode(outputs,
                                                 skip_special_tokens=True)
        definitions += predictions
    logging.info(f"Generating definitions finished")
    return definitions


if __name__ == '__main__':
    args = parse_arge()
    model_path = os.path.join(args.models_dir, MODELS[args.lang])
    if "mt0" in model_path:
        model, tokenizer = load_mt5_model_and_tokenizer(model_path)
    else:
        model, tokenizer = load_model_and_tokenizer(model_path)
    lang_folder = os.path.join(
        os.path.expanduser(args.res_path),
        args.lang,
    )
    if not os.path.isdir(lang_folder):
        os.mkdir(lang_folder)
    res_folder = os.path.join(lang_folder, args.decoding_strategy)
    if not os.path.isdir(res_folder):
        os.mkdir(res_folder)
    if args.decoding_strategy == "greedy":
        assert args.num_beams == 1
        assert args.num_beam_groups == 1
        assert args.diversity_penalty == 0
    elif "beam" in args.decoding_strategy:
        assert args.num_beams > 1
        if "diverse" in args.decoding_strategy:
            assert args.num_beam_groups > 1
    logging.info(f"Will save to {res_folder}")
    for corpus in glob(f"{args.data_path}/{args.lang}/*.gz"):
        logging.info(corpus)
        prompts, targets_list = [], []
        with gzip.open(corpus, "rt", encoding="utf8") as corpus_file:
            count = 0
            for line in tqdm.tqdm(corpus_file):
                if (args.n_first == 0) or (count < args.n_first):
                    target, prompt = line.strip().split('\t')
                    prompts.append(prompt)
                    targets_list.append(target)
                    count += 1
                else:
                    break
        definitions = define(prompts, model, tokenizer, args, targets_list)
        res_fn = corpus.split("/")[-1].split(os.extsep)[0].replace(
            "home-m-corpora-acl-parsed-", "",
        ) + ".tsv.gz"
        res_path = os.path.join(
            res_folder,
            res_fn,
        )
        results = pd.DataFrame({0: targets_list, 1: prompts, 2: definitions})
        logging.info(f"Number of lines in the result: {results.shape[0]}")
        results.to_csv(
            res_path,
            sep="\t",
            index=False,
            compression="gzip",
            header=False,
            encoding="utf8",
        )

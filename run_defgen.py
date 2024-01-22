import argparse
from glob import glob
import gzip
import logging
import os

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
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer(model_path):
    tokenizer = T5Tokenizer.from_pretrained(
        model_path,
        add_prefix_space=True,
    )
    if DEVICE == "cuda":
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
    if DEVICE == "cuda":
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
        choices=LANGUAGES,
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

    test_dataset = torch.utils.data.TensorDataset(
        inputs["input_ids"].to(DEVICE),
        inputs["attention_mask"].to(DEVICE),
        target_ids.to(DEVICE),
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
                outputs = lm.generate(input_ids=inp, attention_mask=att,
                                      do_sample=False, bad_words_ids=bad)
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
    for corpus in glob(f"{args.data_path}/{args.lang}/*.gz"):
        logging.info(corpus)
        prompts, targets_list = [], []
        with gzip.open(corpus, "rt") as corpus_file:
            count = 0
            for line in tqdm.tqdm(corpus_file):
                if (args.n_first is None) or (count < args.n_first):
                    target, prompt = line.strip().split('\t')
                    prompts.append(prompt)
                    targets_list.append(target)
                    count += 1
                else:
                    break
        definitions = define(prompts, model, tokenizer, args, targets_list)
        res_path = os.path.join(
            os.path.expanduser(args.res_path),
            f"{corpus.replace('/', '-')}{os.extsep}txt",
        )
        with open(res_path, "w") as results_file:
            for prompt, definition in zip(prompts, definitions):
                results_file.write(f"{prompt}\t{definition}\n")

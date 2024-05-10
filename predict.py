import os
import sys
import json
import logging
from collections import Counter
from pathlib import Path
from numpy import who 

from rich import print, progress
from rich.logging import RichHandler

import numpy as np

import torch
import sentencepiece as spm

from src.transformer import Transformer
from src.parameters import src_vocab_size, trg_vocab_size, pad_id
from src.utils import pad_or_truncate, process_raw_spectrum, logR
from src.predict import greedy_search, beam_search

logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
        )

logger = logging.getLogger(__name__)


def model_generator(model, tokenized_input):
    logger.debug(f"Running model generator...")

    src_data = torch.LongTensor(pad_or_truncate(tokenized_input)).unsqueeze(0).to(args.device) # (1, L)
    e_mask = (src_data != pad_id).unsqueeze(1).to(args.device) # (1, 1, L)

    logger.debug(f"src input: {len(tokenized_input) = }\n{src_data.shape = }\n{e_mask.shape = }")

    model.eval()
    src_data = model.src_embedding(src_data)
    logger.debug(f"model src_embedding: {src_data.shape = }")
    src_data = model.positional_encoder(src_data)
    logger.debug(f"model positional_encoder: {src_data.shape = }")
    e_output = model.encoder(src_data, e_mask) # (B, L, d_model)
    logger.debug(f"model encoder: {e_output.shape = }")

    if args.decode == 'greedy':
        result = greedy_search(model, e_output, e_mask)
    # elif method == 'beam':
    #     result = beam_search(model, e_output, e_mask)
    logger.debug(f"Decoded Token IDs: {result}")

    return result



def main(args):
    # Model initialization
    model = Transformer(src_vocab_size, trg_vocab_size)

    state_dicts = torch.load(args.checkpoint_path, map_location=args.device)
    if 'model_state_dict' in state_dicts:
        model.load_state_dict(state_dicts['model_state_dict'])
    else:
        model.load_state_dict(state_dicts)
    model.to(args.device)

    # Toekenizer initialization
    src_sp = spm.SentencePieceProcessor(model_file="data/sp/src_sp.model")
    trg_sp = spm.SentencePieceProcessor(model_file="data/sp/trg_sp.model")

    # Prepare src src_data
    input_src_list = []
    if args.preprocess:
        # Currently it expect one spectrum per file
        # raw spectrum file is expected to be in the format of 'mass\tintensity' per line and separated by newlines.
        # input_text = "25i5 28i5 29i4 41i2 52i4 53i3 54i5 57i5 58i4 59i5 65i3 66i5 76i5 77i3 78i4 83i3 94i5 95i5 106i4 107i4 233i1 234i4 248i2 249i4"
        input_text = process_raw_spectrum(
            args.spectrum_file,
            normalize=True,
            intensity_threshold=0.4
        )
        input_text[:, 1] = logR(input_text[:, 1])
        input_text = input_text.astype(int)
        input_text = ' '.join([f'{mz}i{intensity}' for mz, intensity in input_text])
        logger.debug(f"Input src: {input_text}")
        input_src_list.append(input_text)
    else:
        for datapoint in open(args.spectrum_file, 'r'):
            input_src_list.append(datapoint.strip())

    tokenized_list = [
            pad_or_truncate(src_sp.EncodeAsIds(input_text)) for input_text in input_src_list]

    # Inference loop: model generator
    results = []
    for i, tokenized_input in enumerate(tokenized_list):
        logger.info(f"Processing input {i+1}/{len(tokenized_list)}")
        decoded_ids = model_generator(model, tokenized_input)
        result = trg_sp.decode_ids(decoded_ids)
        all_rAEs = set(result.replace(".", " ").replace(" <mol> ", " ").strip().split(" "))
        all_rAEs_count = Counter(result.replace(".", " ").replace(" <mol> ", " ").strip().split(" "))
        mol_rAEs = set(result.split(" <mol> ")[-1].split(" "))
        results.append({
            "src": input_src_list[i],
            "raw_output": result,
            "pred_all_rAEs": list(all_rAEs),
            "pred_all_rAEs_count": dict(all_rAEs_count),
            "pred_mol_rAEs": list(mol_rAEs),
            })

        logger.debug(f"Raw output: {result}")
        logger.debug(f"all_rAEs: {all_rAEs}")
        logger.debug(f"all_rAEs_count: {all_rAEs_count}")
        logger.debug(f"mol_rAEs: {mol_rAEs}")

    logger.info(f"Saving results...{args.output = }")
    json.dump(results, open(args.output, 'w'), indent=4)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--spectrum_file', type=Path, required=True, help="The file path where the input spectra can be found. It expects a list of peaks 'mass\tintensity' delimited by lines.")
    parser.add_argument(
            '-pre', '--preprocess', action='store_true', help="Whether to preprocess the input spectra." )
    parser.add_argument(
            '-o', '--output', type=Path, default='results.json', help="The path to save the results.")
    parser.add_argument(
            '--checkpoint_path', type=Path, default='saved_models/checkpoint.pth', help="The path to the model checkpoint.")
    parser.add_argument(
            '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument(
            '--decode', type=str, default='greedy', choices=['greedy', 'beam'], help= "The decoding method to use during inference. Default is greedy.")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode.")

    args = parser.parse_args()
    logger.info(f"Arguments: {args}")

    if args.debug:
        logger.setLevel(logging.DEBUG)
    main(args)


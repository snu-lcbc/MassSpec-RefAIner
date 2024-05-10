from .parameters import *
from .utils import *
from .transformer import *

from torch import nn
import torch

import sys, os
import argparse
import datetime
import copy
import heapq, json
from pprint import pprint

from rdkit import Chem

import sentencepiece as spm
import numpy as np
import pandas as pd
import multiprocessing as mp


def setup(model, checkpoint_name):
    assert os.path.exists(f"{ckpt_dir}/{checkpoint_name}"), f"There is no checkpoint named {checkpoint_name}."

    print("Loading checkpoint...\n")
    checkpoint = torch.load(f"{ckpt_dir}/{checkpoint_name}")
    model.load_state_dict(checkpoint['model_state_dict'])
    #optim.load_state_dict(checkpoint['optim_state_dict'])
    #best_loss = checkpoint['loss']

    return model

def tanimoto_similarity(s_truth, s_pred):
    lpred = s_pred.replace(' ', ' ').split()#.split()
    ltruth = s_truth.replace(' ', ' ').split()#.split()
    return len(set(ltruth) & set(lpred)) / max(float(len(set(ltruth) | set(lpred))), 1e-6)

def custom_validation_fn(model, test_loader, method='greedy'):
    src_sp = spm.SentencePieceProcessor()
    trg_sp = spm.SentencePieceProcessor()
    src_sp.Load(f"{SP_DIR}/src_sp.model")
    trg_sp.Load(f"{SP_DIR}/trg_sp.model")

    start_time = datetime.datetime.now()
    total, exact, _100, _90, _85, _80, _70, _60, _50, _40, _30, _20, _00, _invalid = 0,0,0, 0,0,0,0,0, 0, 0, 0, 0,0, 0
    fp_tcs = []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            src_input, trg_input, trg_output = batch
            src_input, trg_input, trg_output = src_input.to(device), trg_input.to(device), trg_output.to(device)

            #e_mask, d_mask = make_mask(src_input, trg_input)
            for j in  range(len(src_input)):
                # preparing src data for encoder
                src_j = src_input[j].unsqueeze(0).to(device) # (L) => (1, L)
                encoder_mask = (src_j != pad_id).unsqueeze(1).to(device) # (1, L) => (1, 1, L)
                # encoding src input
                src_j = model.src_embedding(src_j) # (1, L) => (1, L, d_model)
                src_j = model.positional_encoder(src_j) # (1, L, d_model)
                encoder_output = model.encoder(src_j, encoder_mask) # (1, L, d_model)
                if method == 'greedy':
                    # s_pred = greedy_search(model, encoder_output, encoder_mask, trg_sp)
                    s_pred_ids = greedy_search(model, encoder_output, encoder_mask)
                    s_pred = trg_sp.decode_ids(s_pred_ids)
                elif method == 'beam':
                    # s_pred = beam_search(model, encoder_output, encoder_mask, trg_sp)
                    s_pred_ids = beam_search(model, encoder_output, encoder_mask)
                    s_pred = trg_sp.decode_ids(s_pred_ids)


                s_src   = src_sp.decode_ids(src_input[j].tolist())
                s_truth = trg_sp.decode_ids(trg_output[j].tolist())

                # if model_type in ['char', 'atom', 'spe']:
                #     tanimoto = FpSimilarity(s_truth.replace(' ', ''), s_pred.replace(' ', ''), radius=2, nBits=2048)
                # elif model_type in ['ae', 'ae0', 'ae2']:
                #     tanimoto = FpSimilarity(AesDecoder(s_truth), AesDecoder(s_pred), radius=2, nBits=2048)
                # elif model_type in ['ais']:
                #     tanimoto = FpSimilarity(AisDecoder(s_truth), AisDecoder(s_pred), radius=2, nBits=2048)
                tanimoto = tanimoto_similarity(s_truth, s_pred)

                fp_tcs.append(tanimoto)
                #predf.write(f"{aetc}|{fptc}|{s_truth}|{s_pred}\n")
                print(f"Prediction|{tanimoto}|{s_truth}|{s_pred}|{s_src}")

                total += 1
                if s_truth == s_pred:
                    exact += 1
                if tanimoto == 1.0:
                    _100 += 1
                elif tanimoto >= 0.90:
                    _90 += 1
                elif tanimoto >= 0.85:
                    _85 += 1
                elif tanimoto >= 0.80:
                    _80 += 1
                elif tanimoto >= 0.70:
                    _70 += 1
                elif tanimoto >= 0.60:
                    _60 += 1
                elif tanimoto >= 0.50:
                    _50 += 1
                elif tanimoto >= 0.40:
                    _40 += 1
                elif tanimoto >= 0.30:
                    _30 += 1
                elif tanimoto >= 0.20:
                    _20 += 1
                elif tanimoto > 0.0:
                    _00 += 1
                else:
                    _invalid += 1


    end_time = datetime.datetime.now()
    validation_time = end_time - start_time
    seconds = validation_time.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    elapsed_time = f"{hours}hrs {minutes}mins {seconds}secs"

    print(f"{np.mean(fp_tcs) = }")
    print(f"{total, exact/total, exact, _100, _90, _85, _80, _70, _60, _50, _40, _30, _20, _00, _invalid} \t{elapsed_time=}")



def greedy_search(model, e_output, e_mask):
# def greedy_search(model, e_output, e_mask, trg_sp):
    last_words = torch.LongTensor([pad_id] * seq_len).to(device) # (L)
    last_words[0] = sos_id # (L)
    cur_len = 1

    for i in range(seq_len):
        d_mask = (last_words.unsqueeze(0) != pad_id).unsqueeze(1).to(device) # (1, 1, L)
        nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool).to(device)  # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask  # (1, L, L) padding false

        trg_embedded = model.trg_embedding(last_words.unsqueeze(0))
        trg_positional_encoded = model.positional_encoder(trg_embedded)
        decoder_output = model.decoder(
            trg_positional_encoded,
            e_output,
            e_mask,
            d_mask
        ) # (1, L, d_model)

        output = model.softmax(
            model.output_linear(decoder_output)
        ) # (1, L, trg_vocab_size)

        output = torch.argmax(output, dim=-1) # (1, L)
        last_word_id = output[0][i].item()

        if i < seq_len-1:
            last_words[i+1] = last_word_id
            cur_len += 1

        if last_word_id == eos_id:
            break

    if last_words[-1].item() == pad_id:
        decoded_output = last_words[1:cur_len].tolist()
    else:
        decoded_output = last_words[1:].tolist()
    # decoded_output = trg_sp.decode_ids(decoded_output)

    return decoded_output

def beam_search(model, e_output, e_mask):
# def beam_search(model, e_output, e_mask, trg_sp):
    cur_queue = PriorityQueue()
    for k in range(beam_size):
        cur_queue.put(BeamNode(sos_id, -0.0, [sos_id]))

    finished_count = 0

    for pos in range(seq_len):
        new_queue = PriorityQueue()
        for k in range(beam_size):
            node = cur_queue.get()
            if node.is_finished:
                new_queue.put(node)
            else:
                trg_input = torch.LongTensor(node.decoded + [pad_id] * (seq_len - len(node.decoded))).to(device) # (L)
                d_mask = (trg_input.unsqueeze(0) != pad_id).unsqueeze(1).to(device) # (1, 1, L)
                nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool).to(device)
                nopeak_mask = torch.tril(nopeak_mask) # (1, L, L) to triangular shape
                d_mask = d_mask & nopeak_mask # (1, L, L) padding false

                trg_embedded = model.trg_embedding(trg_input.unsqueeze(0))
                trg_positional_encoded = model.positional_encoder(trg_embedded)
                decoder_output = model.decoder(
                    trg_positional_encoded,
                    e_output,
                    e_mask,
                    d_mask
                ) # (1, L, d_model)

                output = model.softmax(
                    model.output_linear(decoder_output)
                ) # (1, L, trg_vocab_size)

                output = torch.topk(output[0][pos], dim=-1, k=beam_size)
                last_word_ids = output.indices.tolist() # (k)
                last_word_prob = output.values.tolist() # (k)

                for i, idx in enumerate(last_word_ids):
                    new_node = BeamNode(idx, -(-node.prob + last_word_prob[i]), node.decoded + [idx])
                    if idx == eos_id:
                        new_node.prob = new_node.prob / float(len(new_node.decoded))
                        new_node.is_finished = True
                        finished_count += 1
                    new_queue.put(new_node)

        cur_queue = copy.deepcopy(new_queue)

        if finished_count == beam_size:
            break

    decoded_output = cur_queue.get().decoded

    if decoded_output[-1] == eos_id:
        decoded_output = decoded_output[1:-1]
    else:
        decoded_output = decoded_output[1:]

    # return trg_sp.decode_ids(decoded_output)
    return decoded_output


def inference(model, input_sentence, model_type,  method):
    src_sp = spm.SentencePieceProcessor()
    trg_sp = spm.SentencePieceProcessor()
    src_sp.Load(f"{SP_DIR}/src_sp.model")
    trg_sp.Load(f"{SP_DIR}/trg_sp.model")

    tokenized = src_sp.EncodeAsIds(input_sentence)
    src_data = torch.LongTensor(pad_or_truncate(tokenized)).unsqueeze(0).to(device) # (1, L)
    e_mask = (src_data != pad_id).unsqueeze(1).to(device) # (1, 1, L)

    start_time = datetime.datetime.now()

    #print("Encoding input sentence...")
    model.eval()
    src_data = model.src_embedding(src_data)
    src_data = model.positional_encoder(src_data)
    e_output = model.encoder(src_data, e_mask) # (1, L, d_model)

    if method == 'greedy':
       # print("Greedy decoding selected.")
        result = greedy_search(model, e_output, e_mask, trg_sp)
    elif method == 'beam':
       # print("Beam search selected.")
        result = beam_search(model, e_output, e_mask, trg_sp)

    end_time = datetime.datetime.now()

    total_inference_time = end_time - start_time
    seconds = total_inference_time.seconds
    minutes = seconds // 60
    seconds = seconds % 60

    #print(f"Input: {input_sentence}")
    #print(f"Result: {result}")
    #print(f"Inference finished! || Total inference time: {minutes}mins {seconds}secs")
    return result


def main(args):
    uni_model = setup(build_model(model_type='uni'), args.uni_checkpoint_name)
    bi_model = setup(build_model(model_type='bi'), args.bi_checkpoint_name)

    print(f'{args.decode} decoding searching method selected')
    print(f"Preprocessing input SMILES: {args.smiles}")
    tokens_list, tokens_str = getAtomEnvs(args.smiles)
    print(f"Atom Envs: {tokens_str}\n")

    if args.smiles:
        assert args.decode == 'greedy' or args.decode =='beam', "Please specify correct decoding method, either 'greedy' or 'beam'."

        r = {}
        r['uni_0'] = inference(uni_model, tokens_str, 'uni', args.decode)
        r['uni_0biR0_R1'], r['uni_0biR0_R2'] = inference(bi_model, r['uni_0'].strip(), 'bi', args.decode).split('.')
        r['uni_0biR1_R1'], r['uni_0biR1_R2'] = inference(bi_model, r['uni_0biR0_R1'].strip(),'bi',  args.decode).split('.')
        r['uni_0biR2_R1'], r['uni_0biR2_R2'] = inference(bi_model, r['uni_0biR0_R2'].strip(), 'bi', args.decode).split('.')

        r['biR0_R1'], r['biR0_R2'] = inference(bi_model, tokens_str, 'bi', args.decode).split('.')
        r['biR1_R1'], r['biR1_R2'] = inference(bi_model, r['biR0_R1'].strip(), 'bi', args.decode).split('.')
        r['biR2_R1'], r['biR2_R2'] = inference(bi_model, r['biR0_R2'].strip(), 'bi', args.decode).split('.')


        result_file = f'./results_{Chem.MolToInchiKey(Chem.MolFromSmiles(args.smiles))}'
        if args.database_dir and os.path.exists(args.database_dir):
            print("\nDatabase searching...")
            results_df = mp_dbSearch(r, args.database_dir)
            print(f'Saving the results here: {result_file}.csv')
            results_df.to_csv(f'{result_file}.csv', index=False)
        else:
            print('There is no database available for retrieval.')
            print("\nModel predicted AEs")
            #print(r)
            print()
            print(f'Saving the results here: {result_file}.json')
            json.dump(r, open(f'{result_file}.json', 'w'))

        print('Done!')

    else:
        print("Please enter input SMILES.")





if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles', type=str, required=True, help='An input sequence')
    parser.add_argument('--decode', type=str, default='greedy', help="greedy or beam?")
    parser.add_argument('--uni_checkpoint_name', type=str, default='uni_checkpoint.pth', help="checkpoint file name")
    parser.add_argument('--bi_checkpoint_name', type=str,  default='bi_checkpoint.pth', help="checkpoint file name")
    parser.add_argument('--database_dir', type=str,  default='PubChem_AEs', help="Database for searching predicted molecules")

    args = parser.parse_args()

    main(args)

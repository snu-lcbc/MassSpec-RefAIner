import os 
import sys 
import json

import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import numpy as np
import pandas as pd
import heapq
import warnings
import re
from pathlib import Path

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import rdkit
from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, DataStructs

from .parameters import *
from .transformer import *


def build_model():
    print(f" molecular model is building...")
    #print("Loading vocabs...")
    src_i2w = {}
    trg_i2w = {}

    with open(f"{SP_DIR}/src_sp.vocab", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        word = line.strip().split('\t')[0]
        src_i2w[i] = word

    with open(f"{SP_DIR}/trg_sp.vocab", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        word = line.strip().split('\t')[0]
        trg_i2w[i] = word

    #print(f"The size of src vocab is {len(src_i2w)} and that of trg vocab is {len(trg_i2w)}.")

    return Transformer(src_vocab_size=len(src_i2w), trg_vocab_size=len(trg_i2w)).to(device)


def make_mask(src_input, trg_input):
    e_mask = (src_input != pad_id).unsqueeze(1)  # (B, 1, L)
    d_mask = (trg_input != pad_id).unsqueeze(1)  # (B, 1, L)

    nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool)  # (1, L, L)
    nopeak_mask = torch.tril(nopeak_mask).to(device)  # (1, L, L) to triangular shape
    d_mask = d_mask & nopeak_mask  # (B, L, L) padding false

    return e_mask, d_mask

class BeamNode():
    def __init__(self, cur_idx, prob, decoded):
        self.cur_idx = cur_idx
        self.prob = prob
        self.decoded = decoded
        self.is_finished = False

    def __gt__(self, other):
        return self.prob > other.prob

    def __ge__(self, other):
        return self.prob >= other.prob

    def __lt__(self, other):
        return self.prob < other.prob

    def __le__(self, other):
        return self.prob <= other.prob

    def __eq__(self, other):
        return self.prob == other.prob

    def __ne__(self, other):
        return self.prob != other.prob

    def print_spec(self):
        print(f"ID: {self} || cur_idx: {self.cur_idx} || prob: {self.prob} || decoded: {self.decoded}")

class PriorityQueue():

    def __init__(self):
        self.queue = []

    def put(self, obj):
        heapq.heappush(self.queue, (obj.prob, obj))

    def get(self):
        return heapq.heappop(self.queue)[1]

    def qsize(self):
        return len(self.queue)

    def print_scores(self):
        scores = [t[0] for t in self.queue]
        print(scores)

    def print_objs(self):
        objs = [t[1] for t in self.queue]
        print(objs)

#################
# Spectral preprocessing 

def process_raw_spectrum(
    spectrum_file, 
    normalize=True, 
    intensity_threshold=0.0
):
    raw_spectrum = open(spectrum_file).read()
    spectrum = []
    for i in raw_spectrum.strip().split('\n'):
        mz, intensity = i.split('\t')
        spectrum.append([mz, intensity])
    spectrum = np.array(spectrum).astype(float)

    if normalize:
        if 100 < max(spectrum[:, 1]) <= 1000:
            spectrum[:, 1] /= 10
        elif max(spectrum[:, 1]) <= 1.0:
            spectrum[:, 1] *= 100
            
    return spectrum[np.where(spectrum[:, 1] >= intensity_threshold)] # 

def logR(intensity):
    intensity_ = intensity.copy()
    dict_i = {i: j for i, j in enumerate(intensity)}
    for (
        rank,
        (index, I),
    ) in enumerate(sorted(dict_i.items(), key=lambda i: i[1], reverse=True), 1):
        logR = int(min(np.log2(rank) + 1, 7))
        intensity_[index] = logR

    return intensity_


#################
# Data loaders

def get_data_loader(file_name):
    src_sp = spm.SentencePieceProcessor()
    trg_sp = spm.SentencePieceProcessor()
    src_sp.Load(f"{SP_DIR}/src_sp.model")
    trg_sp.Load(f"{SP_DIR}/trg_sp.model")

    print(f"Getting source/target {file_name} ...")
    with open(f"{DATA_DIR}/{SRC_DIR}/{file_name}", 'r', encoding="utf-8") as f:
        src_text_list = f.readlines()

    with open(f"{DATA_DIR}/{TRG_DIR}/{file_name}", 'r', encoding="utf-8") as f:
        trg_text_list = f.readlines()

    print("Tokenizing & Padding src data...")
    src_list = process_src(src_text_list, src_sp) # (sample_num, L)
    print(f"The shape of src data: {np.shape(src_list)}")

    print("Tokenizing & Padding trg data...")
    input_trg_list, output_trg_list = process_trg(trg_text_list, trg_sp) # (sample_num, L)
    print(f"The shape of input trg data: {np.shape(input_trg_list)}")
    print(f"The shape of output trg data: {np.shape(output_trg_list)}")

    dataset = CustomDataset(src_list, input_trg_list, output_trg_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def pad_or_truncate(tokenized_text):
    if len(tokenized_text) < seq_len:
        left = seq_len - len(tokenized_text)
        padding = [pad_id] * left
        tokenized_text += padding
    else:
        tokenized_text = tokenized_text[:seq_len]

    return tokenized_text

def process_src(text_list, src_sp):
    tokenized_list = []
    for text in text_list:
        tokenized = src_sp.EncodeAsIds(text.strip())
        tokenized_list.append(pad_or_truncate(tokenized + [eos_id]))

    return tokenized_list

def process_trg(text_list, trg_sp):
    input_tokenized_list = []
    output_tokenized_list = []
    for text in text_list:
        tokenized = trg_sp.EncodeAsIds(text.strip())
        trg_input = [sos_id] + tokenized
        trg_output = tokenized + [eos_id]
        input_tokenized_list.append(pad_or_truncate(trg_input))
        output_tokenized_list.append(pad_or_truncate(trg_output))

    return input_tokenized_list, output_tokenized_list

class CustomDataset(Dataset):
    def __init__(self, src_list, input_trg_list, output_trg_list):
        super().__init__()
        self.src_data = torch.LongTensor(src_list)
        self.input_trg_data = torch.LongTensor(input_trg_list)
        self.output_trg_data = torch.LongTensor(output_trg_list)

        assert np.shape(src_list) == np.shape(input_trg_list), "The shape of src_list and input_trg_list are different."
        assert np.shape(input_trg_list) == np.shape(output_trg_list), "The shape of input_trg_list and output_trg_list are different."

    def make_mask(self):
        e_mask = (self.src_data != pad_id).unsqueeze(1) # (num_samples, 1, L)
        d_mask = (self.input_trg_data != pad_id).unsqueeze(1) # (num_samples, 1, L)

        nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool) # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask) # (1, L, L) to triangular shape
        d_mask = d_mask & nopeak_mask # (num_samples, L, L) padding false

        return e_mask, d_mask

    def __getitem__(self, idx):
        return self.src_data[idx], self.input_trg_data[idx], self.output_trg_data[idx]

    def __len__(self):
        return np.shape(self.src_data)[0]


#################
# Metrics for evaluation

def set_based_recall(y_true, y_pred):
    true_set = set(y_true)
    pred_set = set(y_pred)
    intersection = true_set.intersection(pred_set)
    if len(true_set) == 0:
        return 0 if len(pred_set) > 0 else 1
    return len(intersection) / len(true_set)

def set_based_precision(y_true, y_pred):
    true_set = set(y_true)
    pred_set = set(y_pred)
    intersection = true_set.intersection(pred_set)
    if len(pred_set) == 0:
        return 1  # No false positives; precision is not affected by the absence of predictions
    return len(intersection) / len(pred_set)

def set_based_accuracy(y_true, y_pred):
    true_set = set(y_true)
    pred_set = set(y_pred)
    intersection = true_set.intersection(pred_set)
    union = true_set.union(pred_set)
    if len(union) == 0:
        return 1  # No true or false predictions, perfect match
    return len(intersection) / len(union)

def rAEsTc(s_truth, s_pred):
    lpred = s_pred.replace('.', ' ').split()#.split()
    ltruth = s_truth.replace('.', ' ').split()#.split()
    return len(set(ltruth) & set(lpred)) / max(float(len(set(ltruth) | set(lpred))), 1e-6)
    
def jaccard_index(a, b):
    set_a = set(a)
    set_b = set(b)
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union

def calculate_score(mol_rae, pred_rae):
    gt_score = len(set(mol_rae) & set(pred_rae)) / len(set(mol_rae))
    pred_score =  len(set(mol_rae) & set(pred_rae)) / max(len(set(pred_rae)), 1e-8)
    return gt_score, pred_score

def fp_similarity(truth_smi, pred_smi, metric=Chem.DataStructs.TanimotoSimilarity):
#     RDLogger.DisableLog('rdApp.*')
    try:
        pred_mol = Chem.MolFromSmiles(pred_smi)
        if pred_mol is None:
            return 0
        
        truth_mol = Chem.MolFromSmiles(truth_smi)
        if truth_mol is None:
            return 0
        
    except:
        return 0
    return DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprintAsBitVect(truth_mol,2,nBits=2048), AllChem.GetMorganFingerprintAsBitVect(pred_mol,2,nBits=2048))



#################
# Extract atom environments

def getSubstructSmi(mol,atomID,radius):
    if radius>0:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol,radius,atomID)
        atomsToUse=[]
        for b in env:
            atomsToUse.append(mol.GetBondWithIdx(b).GetBeginAtomIdx())
            atomsToUse.append(mol.GetBondWithIdx(b).GetEndAtomIdx())
        atomsToUse = list(set(atomsToUse))
    else:
        atomsToUse = [atomID]
        env=None
    symbols = []
    for atom in mol.GetAtoms():
        deg = atom.GetDegree()
        isInRing = atom.IsInRing()
        nHs = atom.GetTotalNumHs()
        symbol = '['+atom.GetSmarts()
        if nHs: 
            symbol += 'H'                                                                                                                
            if nHs>1:
                symbol += '%d'%nHs
        if isInRing:
            symbol += ';R'
        else:
            symbol += ';!R'
        symbol += ';D%d'%deg
        symbol += "]"
        symbols.append(symbol)
    try:
        smile = Chem.MolFragmentToSmiles(mol,atomsToUse,bondsToUse=env,allHsExplicit=True, allBondsExplicit=True, rootedAtAtom=atomID)
        smart = Chem.MolFragmentToSmiles(mol,atomsToUse,bondsToUse=env,atomSymbols=symbols, allBondsExplicit=True, rootedAtAtom=atomID)
    except (ValueError, RuntimeError) as ve:
        print('atom to use error or precondition bond error')
        return None, None
    return smile, smart

def getSmiSma(smiles):
    molecule = smiles.strip()
    molP = Chem.MolFromSmiles(molecule)
    if molP is None: 
        return None, None
    sanitFail = Chem.SanitizeMol(molP, catchErrors=True)
    if sanitFail:
        return None, None
    info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(molP,radius=1,nBits=1024,bitInfo=info)

    info_temp = []

    for bitId,atoms in info.items():
        exampleAtom,exampleRadius = atoms[0]
        description = getSubstructSmi(molP,exampleAtom,exampleRadius)
        if description == None:
            print('Error in getSubstructSmi',smiles,molecule)
        info_temp.append((bitId, exampleRadius, description[0], description[1]))
        
     #collect the desired output in another list
    updateInfoTemp = []
    for k,j in enumerate(info_temp):
        if j[1] == 0:
            updateInfoTemp.append(j)
        else:
            continue
        
    smis = []
    smas = []
    for i in updateInfoTemp:
        smis.append(i[2])
        smas.append(i[3])
    return smis, smas


def smiles_tokenizer(smi):
    import re
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    #assert smi == ''.join(tokens)
    #return ' '.join(tokens), len(tokens)
    return tokens



#################
# Mass Spectra Library Search 

def process_spectrum(raw_spectrum, delimeter=':'):
    mz, intensity = [], []
    for ms in raw_spectrum.strip().split():
        m, i = ms.split(delimeter)
        mz.append(int(float(m)))
        intensity.append(float(i))
        
    return np.array(mz), np.array(intensity)

def wtMzI(mz, intensity, m=1, n=0.5, spectra_len = 1000):
    mzI = np.zeros(spectra_len)
    if max(mz) > spectra_len:
        index = np.where(mz < spectra_len)[0]
        mz =  mz[index]
        intensity = intensity[index] 
    mzI[mz] = (mz ** n) * (intensity ** n)
    
    return mzI

def CosSimilarity(i, j):
    return np.dot(i, j) / max((np.linalg.norm(i) * np.linalg.norm(j)), 1e-8)

def cosine_similarity(query_spectrum, ref_spectrum, spectra_len = 1000 ):
    query_mz, query_intensity  = process_spectrum(query_spectrum )
    ref_mz,   ref_intensity  = process_spectrum(ref_spectrum)

    if max(query_mz) > spectra_len:
        idx = np.where(query_mz < spectra_len)[0]
        query_mz = query_mz[idx]
        query_intensity = query_intensity[idx]
        
    if max(ref_mz) > spectra_len:
        idx = np.where(ref_mz < spectra_len)[0]
        ref_mz = ref_mz[idx]
        ref_intensity = ref_intensity[idx]

    query_mzI  = np.zeros(spectra_len )
    query_mzI[query_mz] = query_mz * query_intensity**0.5

    ref_mzI = np.zeros(spectra_len )
    ref_mzI[ref_mz]   = ref_mz * ref_intensity**0.5

    return np.dot(query_mzI, ref_mzI)/max((np.linalg.norm(query_mzI)*np.linalg.norm(ref_mzI)), 1e-8)

def timing(f):
    import time
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        seconds = time2 - time1
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60

        print(f'{f.__name__} -> Elapsed time: {hours}hrs {minutes}mins {seconds:.3}secs')

        return ret
    return wrap


def db_search(query, dbdir, topk):
    query_set = set(query.strip().split())

    resultq = []
    for file in Path(dbdir).iterdir():
        if file.name.endswith('smarts'):
            with open(file, 'r') as fp:
                for i, item in enumerate(fp.readlines(), 1):
                    if i % 1000000 == 0:
                        print(i)
                    sequence = item.strip().split('\t')
                    smiles = sequence[0].strip()
                    aes_str = sequence[1].strip()
                    aes_set = set(sequence[1].strip().split())
                    tanimoto = tc(query_set, aes_set)
                    if tanimoto >= 0.8:
                        #result.append((location, query_noform, smile, nbit_noform, tanimoto))
                        heapq.heappush(resultq, (-tanimoto, query, aes_str, smiles))
    c = 1
    candidates = []
    until = topk if len(resultq) > topk else len(resultq)

    while c <= until:
        c += 1
        try:
            candidates.append(heapq.heappop(resultq))
        except:
            pass
    return candidates


@timing
def mp_dbSearch(results_dict, dbdir, topk=5):
    import multiprocessing as mp
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(mp.cpu_count()+2)
    jobs = []
    for k, v in results_dict.items():
        job = pool.apply_async(db_search, (v, dbdir, topk ))
        jobs.append((k, job))

    results = []
    for k, job in jobs:
        #results.append(job.get())
        candidates = job.get()
        for tanimoto, query, aes_str, smiles in candidates:
            results.append([k, query, aes_str, smiles, -tanimoto])

    #return results
    return pd.DataFrame(results, columns=['Tree', 'Model_Prediction', "DB_AEs", "DB_SMILES", "DB_Tc"])


#################
# Misc

def to_jsonl(element, file_path):
    with open(file_path, 'a') as file:
        json.dump(element, file)
        file.write('\n')

def from_jsonl(file_path):
    with open(file_path,'r') as file:
        for line in file:
            yield json.loads(line)


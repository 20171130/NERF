from rdkit import Chem
import pandas as pd
import numpy as np
import torch
import csv
import os
import pickle
import re
import pdb
from tqdm import tqdm, trange
from concurrent.futures import ProcessPoolExecutor
from rdkit import RDLogger
'''
aroma: [B, L]
e: [B, L]
b: [B, L, 4]
c: [B, L]
m: [B, L]
'''
MAX_BONDS = 6
MAX_DIFF = 4
prefix = "data"

def molecule(mols, src_len, reactant_mask = None, ranges = None):
    features = {}
    element = np.zeros(src_len, dtype='int32')
    aroma = np.zeros(src_len, dtype='int32')
    bonds = np.zeros((src_len, MAX_BONDS), dtype='int32')
    charge = np.zeros(src_len, dtype='int32')
    
    reactant = np.zeros(src_len, dtype='int32') # 1 for reactant
    mask = np.ones(src_len, dtype='int32') # 1 for masked
    segment = np.zeros(src_len, dtype='int32')

    for molid, mol in enumerate(mols):
        for atom in mol.GetAtoms():
            idx = atom.GetAtomMapNum()-1

            segment[idx] = molid
            element[idx] = atom.GetAtomicNum()
            charge[idx] = atom.GetFormalCharge()
            mask[idx] = 0
            if reactant_mask:
                reactant[idx] = reactant_mask[molid]

            cnt = 0
            for j, b in enumerate(atom.GetBonds()): # mark existence of bond first
                other = b.GetBeginAtomIdx() + b.GetEndAtomIdx() - atom.GetIdx()
                other = mol.GetAtoms()[other].GetAtomMapNum() - 1
                num_map = {'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3, 'AROMATIC': 1}
                num = num_map[str(b.GetBondType())]
                for k in range(num):
                    if cnt == MAX_BONDS:
                        return None
                    bonds[idx][cnt] = other
                    cnt += 1 
                if str(b.GetBondType()) == 'AROMATIC':
                    aroma[idx] = 1
            tmp = bonds[idx][0:cnt]
            tmp.sort()
            bonds[idx][0:cnt] = tmp
            while cnt < MAX_BONDS:
                bonds[idx][cnt] = idx
                cnt += 1
            
    features = {'element':element, 'bond':bonds, 'charge':charge, 'aroma':aroma, 'mask':mask, 'segment':segment, 'reactant': reactant}
    return features


def reaction(args):
    """ processes a reaction, returns dict of arrays"""
    src, tgt = args
    pattern = re.compile(":(\d+)\]") # atom map numbers
    src_len = Chem.MolFromSmiles(src).GetNumAtoms()

    # reactant mask
    src_mols = src.split('.')
    tgt_atoms = pattern.findall(tgt)
    reactant_mask = [False for i in src_mols]
    for j, item in enumerate(src_mols):
        atoms = pattern.findall(item)
        for atom in atoms:
            if atom in tgt_atoms:
                reactant_mask[j] = True
                break  
                
    # the atom map num ranges of each molecule for segment mask
    src_mols = [Chem.MolFromSmiles(item) for item in src_mols]
    tgt_mols = [Chem.MolFromSmiles(item) for item in tgt.split(".")]
    ranges = []
    for mol in src_mols:
        lower = 999
        upper = 0
        for atom in mol.GetAtoms():
            lower = min(lower, atom.GetAtomMapNum()-1)
            upper = max(upper, atom.GetAtomMapNum())
        ranges.append((lower, upper))
    
    src_features = molecule(src_mols, src_len, reactant_mask, ranges)
    tgt_features = molecule(tgt_mols, src_len)
    
    
    if not (src_features and tgt_features):
        return None
                
    src_bond = src_features['bond']
    tgt_bond = tgt_features['bond']
    bond_inc = np.zeros((src_len, MAX_DIFF), dtype='int32')
    bond_dec = np.zeros((src_len, MAX_DIFF), dtype='int32')
    for i in range(src_len):
        if tgt_features['mask'][i]:
            continue
        inc_cnt = 0
        dec_cnt = 0
        diff = [0 for _ in range(src_len)]
        for j in range(MAX_BONDS):
            diff[tgt_bond[i][j]] += 1
            diff[src_bond[i][j]] -= 1
        for j in range(src_len):
            if diff[j] > 0:
                if inc_cnt + diff[j] >MAX_DIFF:
                    return None
                bond_inc[i][inc_cnt:inc_cnt+diff[j]] = j
                inc_cnt += diff[j]
            if diff[j] < 0:
                bond_dec[i][dec_cnt:dec_cnt-diff[j]] = j
                dec_cnt -= diff[j]
        assert inc_cnt == dec_cnt
        
    item = {}
    for key in src_features:
        if key in ["element", "reactant"]:
            item[key] = src_features[key]
        else:
            item['src_' + key] = src_features[key]
            item['tgt_' + key] = tgt_features[key]
    return item


def process(name):
    tgt = []
    src = []
    with open(name + ".txt") as file:
        for line in file:
            rxn = line.split()[0].split('>>')
            src.append(rxn[0])
            tgt.append(rxn[1])

    pool = ProcessPoolExecutor(10)
    dataset = []   
    batch_size = 2048
    for i in trange(len(src)//batch_size+1):
        upper = min((i+1)*batch_size, len(src))
        arg_list = [(src[idx], tgt[idx]) for idx in range(i*batch_size, upper)]
        result = pool.map(reaction, arg_list, chunksize= 64)
        result = list(result)  
        for item in result:
            if not item is None:
                dataset += [item]        
    pool.shutdown()

    with open(name +"_"+prefix+ '.pickle', 'wb') as file:
        pickle.dump(dataset, file)
    print("total %d, legal %d"%(len(src), len(dataset)))
    print(name, 'file saved.')

if __name__ =='__main__':
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    RDLogger.DisableLog('rdApp.info') 
    process("data/valid")
    process("data/test")
    process("data/train")
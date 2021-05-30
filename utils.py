from rdkit.Chem import Draw
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
import torch
import os
import json
import pdb

            
def mol2array(mol):
    img = Draw.MolToImage(mol, kekulize=False)
    array = np.array(img)[:, :, 0:3]
    return array

def check(smile):
    smile = smile.split('.')
    smile.sort(key = len)
    try:
        mol = Chem.MolFromSmiles(smile[-1], sanitize=False)
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True
    except Exception:
        return False

def mol2file(m, name):
    AllChem.Compute2DCoords(m)
    img = Draw.MolToImage(m)
    Draw.MolToFile(m, os.path.join('./img', name))


def result2mol(args): # for threading
    element, mask, bond, aroma, charge, reactant = args
    # [L], [L], [L, 4], [l], [l]
    mask = mask.ne(1)
    cur_len = sum(mask.long())
    l = element.shape[0]

    mol = Chem.RWMol()
    
    element = element.cpu().numpy().tolist()
    charge = charge.cpu().numpy().tolist()
    bond = bond.cpu().numpy().tolist()    
    
    # add atoms to mol and keep track of index
    node_to_idx = {}
    for i in range(l):
        if mask[i] == False:
            continue
        a = Chem.Atom(element[i])
        if not reactant is None and reactant[i]:
            a.SetAtomMapNum(i+1)
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx

    # add bonds between adjacent atoms
    for this in range(l):
        if mask[this] == False:
            continue
        lst = bond[this]
        for j in range(len(bond[0])):
            other = bond[this][j]
            # only traverse half the matrix
            if other >= this or other in lst[0:j] or not this in bond[other]:
                continue
            if lst.count(other)==3 or bond[other].count(this) == 3:
                bond_type = Chem.rdchem.BondType.TRIPLE
                mol.AddBond(node_to_idx[this], node_to_idx[other], bond_type) 
            elif lst.count(other) == 2 or bond[other].count(this) == 2:
                bond_type = Chem.rdchem.BondType.DOUBLE
                mol.AddBond(node_to_idx[this], node_to_idx[other], bond_type)   
            else:
                if aroma[this]==aroma[other] and aroma[this]>0: 
                    bond_type = Chem.rdchem.BondType.AROMATIC
                else:
                    bond_type = Chem.rdchem.BondType.SINGLE
                mol.AddBond(node_to_idx[this], node_to_idx[other], bond_type)
                 
    for i, item in enumerate(charge):
        if mask[i] == False:
            continue
        if not item == 0:
            atom = mol.GetAtomWithIdx(node_to_idx[i])
            atom.SetFormalCharge(item)
    # Convert RWMol to Mol object
    mol = mol.GetMol() 
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
    smile = Chem.MolToSmiles(mol)
    return mol, smile, check(smile)

def visualize(element, mask, bond, aroma, charge, reactant=None):
    mol, smile, _ = result2mol((element, mask, bond, aroma, charge, reactant))
    array = mol2array(mol)
    return array, smile
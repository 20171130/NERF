from torch import nn
import torch
import torch.nn.functional as F
import math
import pdb
import os
from torch.nn import MultiheadAttention
from torch.distributions.multinomial import Multinomial

MAX_BONDS = 6
MAX_DIFF = 4

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings55 have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/dim))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/dim))
        \text{where pos is the word position and i is the embed idx)
    Args:
        dim: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    """
    def __init__(self, dim, dropout=0.1, max_len = 150):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.pe = nn.Parameter(pe) # trainable

    def forward(self, l):
        r"""
        returns the additive embedding, notice that addition isnot done in this function
        input shape [l, b, ...] outputshape [l, 1, dim]
        """
        tmp = self.pe[:l, :]
        return self.dropout(tmp)

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, 4*dim, 1)
        self.conv2 = nn.Conv1d(4*dim, dim, 1)
        self.conv3 = nn.Conv1d(dim, 4*dim, 1)
        self.conv4 = nn.Conv1d(4*dim, dim, 1)
        self.conv5 = nn.Conv1d(dim, dim, 1)
        
    def forward(self, x):
        inter = self.conv1(x)
        inter = F.relu(inter)
        inter = self.conv2(inter)
        x = x + inter
        
        inter = self.conv3(x)
        inter = F.relu(inter)
        inter = self.conv4(inter)
        x = x + inter
        
        return self.conv5(x)

    
    
class AtomEncoder(nn.Module):
    def __init__(self, ntoken, dim, dropout=0.1, rank=0):
        super().__init__()
        self.position_embedding = PositionalEncoding(dim, dropout=dropout)
        self.element_embedding = nn.Embedding(ntoken, dim)
        self.charge_embedding = nn.Embedding(13, dim) #[-6, +6]
        self.aroma_embedding = nn.Embedding(2, dim)
        self.reactant_embedding = nn.Embedding(2, dim)
        self.segment_embedding = nn.Embedding(30, dim)
        self.rank = rank
        self.mlp = MLP(dim)
        
    def forward(self, element, bond, aroma, charge, segment, reactant_mask=None):
        '''
        element, long [b, l] element index
        bonds, long [b, l, MAX_BONDS]
        aroma, long [b, l]
        charge, long [b, l] +2 +1 0 -1 -2
        
        returns [l, b, dim]
        
        '''
        b, l = element.shape
        # basic information
        element = element.transpose(1, 0) 
        element_embedding = self.element_embedding(element)
        embedding = element_embedding
        #[l, b, dim]

        position_embedding = self.position_embedding(l)
        embedding = embedding + position_embedding
        
        aroma = aroma.transpose(1, 0).long()
        aroma_embedding = self.aroma_embedding(aroma)
        embedding = embedding + aroma_embedding
        
        # additional information
        charge = charge.transpose(1, 0) + 6  
        charge_embedding = self.charge_embedding(charge)
        embedding = embedding + charge_embedding
        
        segment = segment.transpose(1, 0) 
        segment_embedding = self.segment_embedding(segment)
        embedding = embedding + segment_embedding
        
        if not reactant_mask is None:
            reactant_mask = reactant_mask.transpose(1, 0) 
            reactant_embedding = self.reactant_embedding(reactant_mask)
            embedding = embedding + reactant_embedding  
            
        message = self.mlp(embedding.permute(1, 2, 0)).permute(2, 0, 1)
        eye = torch.eye(l).to(self.rank)
        tmp = torch.index_select(eye, dim=0, index=bond.reshape(-1)).view(b, l, MAX_BONDS, l).sum(dim=2) # adjacenct matrix
        tmp = tmp*(1-eye) # remove self loops
        message = torch.einsum("lbd,bkl->kbd", message, tmp)
        
        embedding = embedding + message
        
        return embedding


class BondDecoder(nn.Module):
    def __init__(self, dim, rank=0):
        super().__init__()
        self.inc_attention = MultiheadAttention(dim, MAX_DIFF)
        self.inc_q = nn.Conv1d(dim, dim, 1)
        self.inc_k = nn.Conv1d(dim, dim, 1)
        
        self.dec_attention = MultiheadAttention(dim, MAX_DIFF)
        self.dec_q = nn.Conv1d(dim, dim, 1)
        self.dec_k = nn.Conv1d(dim, dim, 1)

        self.rank = rank

    def forward(self, molecule_embedding, src_bond, src_mask, tgt_bond=None, tgt_mask=None):
        """
            mask == True iff masked
            molecule_embedding of shape [l, b, dim]
        """
        l, b, dim = molecule_embedding.shape
        molecule_embedding = molecule_embedding.permute(1, 2, 0)  # to [b, dim, l]
        
        q, k, v = self.inc_q(molecule_embedding), self.inc_k(molecule_embedding), molecule_embedding
        q, k, v = q.permute(2, 0, 1), k.permute(2, 0, 1), v.permute(2, 0, 1)  # to [l, b, c]
        _, inc = self.inc_attention(q, k, v, key_padding_mask=src_mask)

        q, k, v = self.dec_q(molecule_embedding), self.dec_k(molecule_embedding), molecule_embedding
        q, k, v = q.permute(2, 0, 1), k.permute(2, 0, 1), v.permute(2, 0, 1)  # to [l, b, c]
        _, dec = self.dec_attention(q, k, v, key_padding_mask=src_mask)
        
        pad_mask = 1 - src_mask.float()
        # [B, L], 0 if padding
        pad_mask = torch.einsum("bl,bk->blk", pad_mask, pad_mask)
        diff = (inc - dec)*MAX_DIFF*pad_mask
        
        eye = torch.eye(src_mask.shape[1]).to(self.rank)
        src_weight = torch.index_select(eye, dim=0, index=src_bond.reshape(-1)).view(b, l, MAX_BONDS, l).sum(dim=2)* pad_mask
        pred_weight = src_weight + diff      
        
        if tgt_bond is None: # inference
            # [b, l, l]
            bonds = []
            pred_weight = (pred_weight + pred_weight.permute(0, 2, 1))/2
            for i in range(MAX_BONDS):
                bonds += [pred_weight.argmax(2)]
                pred_weight -= torch.index_select(eye, dim=0, index=bonds[-1].reshape(-1)).view(b, l, l)
            pred_bond = torch.stack(bonds, dim =2)
            return pred_bond
            
        else: # training
            tgt_mask = tgt_mask.float() # 1 iff masked
            or_mask = 1 - torch.einsum("bl,bk->blk", tgt_mask, tgt_mask) # notice that this doesn't mask the edges between target and side products
            and_mask = torch.einsum("bl,bk->blk", 1-tgt_mask, 1-tgt_mask)
        
            tgt_weight = torch.index_select(eye, dim=0, index=tgt_bond.reshape(-1)).view(b, l, MAX_BONDS, l).sum(dim=2)*and_mask
            error = pred_weight - tgt_weight
            error = error*error*pad_mask*or_mask
            loss = error.sum(dim=(1, 2))
            return {'bond_loss':loss}

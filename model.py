from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules import TransformerDecoder, TransformerDecoderLayer
from layer import AtomEncoder, BondDecoder
from torch import nn
import torch
import torch.nn.functional as F
import math
import pdb
import os


class MoleculeEncoder(nn.Module):
    def __init__(self, ntoken, dim, nhead, nlayer, dropout, rank):
        super().__init__()
        self.atom_encoder = AtomEncoder(ntoken, dim, dropout=dropout, rank=rank)
        layer = TransformerEncoderLayer(dim, nhead, dim, dropout)
        self.transformer_encoder = TransformerEncoder(layer, nlayer)
        # multihead attention assumes [len, batch, dim]
        # padding_mask = True equivalent to mask = -inf

    def forward(self, element, bond, aroma, charge, mask, segment, reactant=None):
        '''
        element, long [b, l] element index
        bonds, long [b, l, 4]
        aroma, long [b, l]
        charge, long [b, l] +1 0 -1
        mask, [b, l] true if masked
        returns [l, b, dim]
        '''
        embedding = self.atom_encoder(element, bond, aroma, charge, segment, reactant)

        encoder_output = self.transformer_encoder(embedding, src_key_padding_mask=mask)
        return encoder_output


class VariationalEncoder(nn.Module):
    def __init__(self, dim, nhead, nlayer, dropout, rank=0):
        super().__init__()
        layer = TransformerDecoderLayer(dim, nhead, dim, dropout)
        self.transformer_decoder = TransformerDecoder(layer, nlayer)
        self.head = nn.Linear(dim, 2*dim)

    def KL(self, posterior):
        # prior is standard gaussian distribution
        mu, logsigma = posterior['mu'], posterior['logsigma']
        # no matter what shape
        logvar = logsigma*2
        loss = 0.5 * torch.sum(mu * mu+ torch.exp(logvar) - 1 - logvar, 1)
        return loss

    def forward(self, src, src_mask, tgt, tgt_mask):
        """
        src, tgt [L, b, dim]
        src_mask, tgt_mask, [B, L]
        """
        l, b, dim = src.shape
        src_mask, tgt_mask = src_mask.permute(0, 1), tgt_mask.permute(0, 1)
        decoder_output = self.transformer_decoder(src, tgt,
                                                  memory_key_padding_mask=tgt_mask, tgt_key_padding_mask=src_mask).permute(1, 2, 0)
        # [L, B, dim] to [B, dim, L]
        tmp = decoder_output * (1-src_mask.float().unsqueeze(1))
        tmp = tmp.mean(dim=2)
        # [B, dim]
        posterior = self.head(tmp)
        result = {}
        result['mu'] = posterior[:, 0:dim]
        result['logsigma'] = posterior[:, dim:]
        return result, self.KL(result)


class MoleculeDecoder(nn.Module):
    def __init__(self, vae, dim, nhead, nlayer, dropout, rank=0):
        super().__init__()
        layer = TransformerEncoderLayer(dim, nhead, dim, dropout)
        self.transformer_encoder = TransformerEncoder(layer, nlayer)
        self.latent_head = nn.Linear(dim, dim)
        self.bond_decoder = BondDecoder(dim, rank)
        self.charge_head = nn.Conv1d(dim, 13, 1) #-6 to +6
        self.aroma_head = nn.Conv1d(dim, 1, 1)
        self.vae = vae
        self.rank = rank

    def forward(self, src, src_bond, src_mask, latent, tgt_bond, tgt_aroma, tgt_charge, tgt_mask):
        l, b, dim = src.size()
        if self.vae:
            tmp = torch.randn(b, dim).to(self.rank)
            latent = tmp * latent['logsigma'].exp() + latent['mu']
            latent = self.latent_head(latent)
            src = src + latent.expand(l, b, dim)

        encoder_output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        eps = 1e-6
        result = self.bond_decoder(encoder_output, src_bond, src_mask, tgt_bond, tgt_mask)

        tgt_mask = 1-tgt_mask.float()
        encoder_output = encoder_output.permute(1, 2, 0)

        aroma_logit = self.aroma_head(encoder_output)
        BCE = nn.BCEWithLogitsLoss(reduction='none')
        tgt_aroma = tgt_aroma.bool().float()
        aroma_logit = aroma_logit.view(b, l)
        aroma_loss = BCE(aroma_logit, tgt_aroma.float()) #[B, L]
        aroma_loss = aroma_loss * tgt_mask
        aroma_loss = aroma_loss.sum(dim=1)
        result['aroma_loss'] = aroma_loss

        charge_logit = self.charge_head(encoder_output)
        CE = nn.CrossEntropyLoss(reduction='none')
        # assumes [B, C, L] (input, target)        
        tgt_charge = tgt_charge.long() + 6
        charge_loss = CE(charge_logit, tgt_charge)
        charge_loss = charge_loss * tgt_mask
        charge_loss = charge_loss.sum(dim=1)
        result['charge_loss'] = charge_loss
        result['pred_loss'] = result['bond_loss'] + aroma_loss + charge_loss
        return result

    def sample(self, src_embedding, src_bond, padding_mask, temperature=1):
        """
            decode the molecule into bond [B, L, 4], given representation of [L, b, dim]
        """
        l, b, dim = src_embedding.shape

        latent = 0
        if self.vae:
            latent = torch.randn(1, b, dim).to(self.rank) *temperature
            latent = self.latent_head(latent)
        src_embedding = src_embedding + latent
        encoder_output = self.transformer_encoder(src_embedding, src_key_padding_mask=padding_mask)
        result = {}
        bond = self.bond_decoder(encoder_output, src_bond, padding_mask)
        result['bond'] = bond.long()
        encoder_output = encoder_output.permute(1, 2, 0)
        # to [b, c, l]
        aroma_logit = self.aroma_head(encoder_output)
        aroma = (aroma_logit > 0).view(b, l)
        result['aroma'] = aroma.long()

        charge_logit = self.charge_head(encoder_output)
        charge = torch.argmax(charge_logit, dim=1)- 6
        result['charge'] = charge.long()

        return result


class MoleculeVAE(nn.Module):
    def __init__(self, args, ntoken, dim=128, nlayer=8, nhead=8, dropout=0.1):
        super().__init__()
        self.args = args
        self.rank = args.local_rank
        self.M_encoder = MoleculeEncoder(ntoken, dim, nhead, nlayer, dropout, self.rank)
        self.P_encoder = MoleculeEncoder(ntoken, dim, nhead, nlayer, dropout, self.rank)
        if args.vae:
            self.V_encoder = VariationalEncoder(dim, nhead, nlayer, dropout, self.rank)
        self.M_decoder = MoleculeDecoder(args.vae, dim, nhead, nlayer, dropout, self.rank)

    def forward(self, mode, tensors, temperature = 1):

        src = self.M_encoder(tensors['element'], tensors['src_bond'], tensors['src_aroma'],
                             tensors['src_charge'], tensors['src_mask'], tensors['src_segment'], tensors['reactant'] )
        if mode is 'train':
            bond, aroma, charge = tensors['tgt_bond'], tensors['tgt_aroma'], tensors['tgt_charge']
            if self.args.vae:

                tgt = self.P_encoder(tensors['element'], bond, aroma, charge,
                                     tensors['tgt_mask'], tensors['tgt_segment'])
                posterior, kl = self.V_encoder(src, tensors['src_mask'], tgt, tensors['tgt_mask'])
                result = self.M_decoder(src, tensors['src_bond'], tensors['src_mask'], posterior,
                                        bond, aroma, charge, tensors['tgt_mask'])
                result['kl'] = kl
                result['loss'] = result['pred_loss'] + self.args.beta*kl
            else:
                result = self.M_decoder(src, tensors['src_mask'], None,
                                        bond, aroma, charge, tensors['tgt_mask'])
                result['loss'] = result['pred_loss'] 

        elif mode is 'sample':
            """ returns bond[B, L, 4], aroma [B, L], charge[B, L], weight [B, L, L]"""
            result = self.M_decoder.sample(src, tensors['src_bond'], tensors['src_mask'], temperature)
        return result


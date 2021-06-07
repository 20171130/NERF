# coding: utf-8

from time import time
import numpy as np
import math
import os
import sys
import json
from tqdm import tqdm, trange
import pdb
import rdkit
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from time import sleep
from model import MoleculeVAE
from dataset import TransformerDataset
from config import parser
from utils import visualize, result2mol
import os
import pickle
import torch.distributed as dist
from concurrent.futures import ProcessPoolExecutor
from rdkit import RDLogger
from optimization import WarmupLinearSchedule

class Trainer(object):
    def __init__(self, dataloader_tr, dataloader_te, dataloader_val, args):
        self.dataloader_tr = dataloader_tr
        self.dataloader_te = dataloader_te
        self.dataloader_val = dataloader_val
        self.args = args
        self.rank = args.local_rank

        self.ntoken = 100
        self.step = 0
        self.eval_step = 0
        self.epoch = 0
        self.epoch_loss = 100

        self.model = MoleculeVAE(args, self.ntoken, args.dim, args.depth).to(self.rank)

           
        self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.rank], find_unused_parameters=True)     
    
        if not self.dataloader_tr is None:
            self._optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                         lr=self.args.lr) 
            self.scheduler = WarmupLinearSchedule(self._optimizer, warmup_steps=5000, t_total=args.epochs*len(self.dataloader_tr))
        if args.checkpoint is not None:
            self.initialize_from_checkpoint()
            
        self.logger = None
        if self.rank is 0:
            self.logger = SummaryWriter("log/"+args.name)
            self.logger.add_text('args', str(self.args), 0)

        
    def initialize_from_checkpoint(self):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        state_dict = {}
        for checkpoint in self.args.checkpoint:
            checkpoint = torch.load(os.path.join(self.args.save_path, checkpoint), map_location=map_location)
            for key in checkpoint['model_state_dict']:
                if key in state_dict:
                    state_dict[key] += checkpoint['model_state_dict'][key]
                else:
                    state_dict[key] = checkpoint['model_state_dict'][key]
        for key in state_dict:
            state_dict[key] = state_dict[key]/len(self.args.checkpoint)
        self.model.module.load_state_dict(state_dict)    
        if self.dataloader_tr is not None:
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        print('initialized!')

    def save_model(self):
        torch.save({
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
        }, args.save_path + 'epoch-' + str(self.epoch) + '-loss-' + str(np.float(self.epoch_loss)))

    def fit(self):
        t_total = time()
        total_loss = []
        print('start fitting')

        for _ in range(self.args.epochs):
            if self.args.eval:
                acc = trainer.evaluate()
                print('epoch %d eval_acc: %.4f' % (self.epoch, acc))
            if self.rank is 0:
                if self.args.save:
                    self.save_model()
            epoch_loss = self.train_epoch(self.epoch)
            self.epoch_loss = epoch_loss
            print('training loss %.4f' % epoch_loss)
            total_loss.append(epoch_loss)

            self.epoch += 1
                
        print('optimization finished ')
        print('Total tiem elapsed: {:.4f}s'.format(time() - t_total))


    def train_epoch(self, epoch_cnt):
        batch_losses = []
        cnt = 0
        true_cnt = 0
        self.model.train()
        torch.cuda.empty_cache()
        if self.rank is 0:
            pbar =  tqdm(self.dataloader_tr)
        else:
            pbar = self.dataloader_tr
        for batch_data in pbar:
            self._optimizer.zero_grad()
            for key in batch_data:
                batch_data[key] = batch_data[key].to(self.rank)
            
            output_dict = self.model('train', batch_data)
            bond_loss = output_dict['bond_loss'].mean()
            aroma_loss = output_dict['aroma_loss'].mean()
            charge_loss = output_dict['charge_loss'].mean()

            if self.rank is 0:
                pbar.set_postfix(n=self.args.name, c='{:.2f}'.format(charge_loss),
                                 a='{:.2f}'.format(aroma_loss), b='{:.2f}'.format(bond_loss))
            loss = output_dict['loss'].mean()
            batch_losses.append(loss.item())
            loss.backward()
            self._optimizer.step()
            self.scheduler.step()
            
            if self.step % 100 == 0 and self.logger:
                self.logger.add_scalar('loss/total', loss.item(), self.step)
                self.logger.add_scalar('loss/bond_loss', bond_loss.item(), self.step)
                self.logger.add_scalar('loss/aroma_loss', aroma_loss.item(), self.step)
                self.logger.add_scalar('loss/charge_loss', charge_loss.item(), self.step)
                if self.args.vae:
                    kl_loss = output_dict['kl'].mean()
                    self.logger.add_scalar('loss/kl_loss', kl_loss.item(), self.step)
                    if kl_loss < 0.5:
                        self.args.beta *= 0.9
                    if kl_loss > 1:
                        self.args.beta *= 1.1
            
            if self.step % 500 == 0:
                output_dict = self.model('sample', batch_data, temperature = 0)
                pred_aroma, pred_charge = output_dict['aroma'], output_dict['charge']
                pred_bond = output_dict['bond']
                element = batch_data['element']
                reactant = batch_data['reactant']
                src_bond, tgt_bond = batch_data['src_bond'], batch_data['tgt_bond']
                tgt_aroma, tgt_charge = batch_data['tgt_aroma'].bool().long(), batch_data['tgt_charge']
                src_mask, tgt_mask = batch_data['src_mask'], batch_data['tgt_mask']
                for j in range(element.size()[0]):
                    _, pred_s, pred_valid = result2mol((element[j], src_mask[j], pred_bond[j], pred_aroma[j],
                                                       pred_charge[j], reactant[j]))
                    _, tgt_s, tgt_valid = result2mol((element[j], tgt_mask[j], tgt_bond[j], tgt_aroma[j], tgt_charge[j], reactant[j]))
                    if tgt_s in pred_s:
                        true_cnt += 1
                cnt += element.size()[0]

            self.step += 1
        if not cnt is 0: # large batch, infrequent sampling
            acc = true_cnt / cnt
        else:
            acc = -1
        print('train acc', acc)
        if self.logger:
            self.logger.add_scalar('acc/train_acc', acc, self.epoch)
        epoch_loss = np.mean(np.array(batch_losses, dtype=np.float))
        return epoch_loss
    
    def evaluate(self):
        true_cnt = 0
        cnt = 0
        pool = ProcessPoolExecutor(2)
        with torch.no_grad():
            self.model.eval()
            pbar = tqdm(self.dataloader_val)
            for batch in iter(pbar):
                self.eval_step += 1
                if not self.eval_step % 3 == 0:
                    continue
                batch_gpu = {}
                for key in batch:
                    batch_gpu[key] = batch[key].to(self.rank)
                b, l = batch['element'].shape
                cnt += b
                tgts = []
                element = batch['element']
                reactant = batch['reactant']
                src_bond, tgt_bond = batch['src_bond'], batch['tgt_bond']
                tgt_aroma, tgt_charge = batch['tgt_aroma'].bool().long(), batch['tgt_charge']
                src_mask, tgt_mask = batch['src_mask'], batch['tgt_mask']
                src_aroma, src_charge = batch['src_aroma'].bool().long(), batch['src_charge']
                
                arg_list = [(element[i], tgt_mask[i], tgt_bond[i], tgt_aroma[i], tgt_charge[i], None) for i in range(b)]
                result = pool.map(result2mol, arg_list, chunksize= 16)
                result = list(result)
                tgts = [item[1].split(".") for item in result] #  _, tgt_s, tgt_valid 

                temperature = 0.7
            
                output_dict = self.model('sample', batch_gpu, temperature)
                pred_aroma, pred_charge = output_dict['aroma'].cpu(), output_dict['charge'].cpu()
                pred_bond = output_dict['bond'].cpu()
                arg_list = [(element[j], src_mask[j], pred_bond[j], pred_aroma[j], pred_charge[j], None) for j in range(b)]
                result = pool.map(result2mol, arg_list, chunksize= 16)
                result = list(result)
                pred_smiles = [item[1].split(".") for item in result] #  _, tgt_s, tgt_valid 
                for j in range(b):
                    # iterate over the batch
                    flag = True
                    for item in tgts[j]:
                        if not item in pred_smiles[j]:
                            flag = False
                    if flag:
                        true_cnt += 1
                        
            idx = np.random.randint(b)
            src, src_smile = visualize(element[idx], src_mask[idx], src_bond[idx], src_aroma[idx], src_charge[idx], reactant[idx])
            pred, pred_smile = visualize(element[idx], src_mask[idx], pred_bond[idx], pred_aroma[idx], pred_charge[idx], reactant[idx])
            tgt, tgt_smile = visualize(element[idx], tgt_mask[idx], tgt_bond[idx], tgt_aroma[idx], tgt_charge[idx], reactant[idx])
            ground_truth = np.concatenate([src, tgt], axis=1)
            pred = np.concatenate([src, pred], axis=1)
            image = np.concatenate([ground_truth, pred], axis=0)
            if self.logger:
                self.logger.add_image('image', image, self.epoch, dataformats='HWC')
                self.logger.add_text('src/tgt/pred', src_smile+">>"+tgt_smile+"//"+pred_smile, self.epoch)
        if not cnt is 0:
            acc = true_cnt / cnt
            print('eval acc %.4f' % acc)
            if self.logger:
                self.logger.add_scalar('acc/accuracy', acc, self.epoch)
            return acc
        else:
            return 0

    def test(self, temperature):
        true_cnt = 0
        cnt = 0    
        valid_cnt = 0
        pool = ProcessPoolExecutor(10)
        test_result = {'temperature':temperature}
        pred  = []
        with torch.no_grad():
            self.model.eval()
            pbar = tqdm(self.dataloader_te)
            for batch in iter(pbar):
                batch_gpu = {}
                for key in batch:
                    batch_gpu[key] = batch[key].to(self.rank)
                b, l = batch['element'].shape
                cnt += b
                tgts = []
                element = batch['element']
                src_bond, tgt_bond = batch['src_bond'], batch['tgt_bond']
                tgt_aroma, tgt_charge = batch['tgt_aroma'].bool().long(), batch['tgt_charge']
                src_mask, tgt_mask = batch['src_mask'], batch['tgt_mask']
                src_aroma, src_charge = batch['src_aroma'].bool().long(), batch['src_charge']
                
                arg_list = [(element[i], tgt_mask[i], tgt_bond[i], tgt_aroma[i], tgt_charge[i], None) for i in range(b)]
                result = pool.map(result2mol, arg_list, chunksize= 64)
                result = list(result)
                tgts = [item[1].split(".") for item in result] #  _, tgt_s, tgt_valid 

                output_dict = self.model('sample', batch_gpu, temperature)
                pred_aroma, pred_charge = output_dict['aroma'].cpu(), output_dict['charge'].cpu()
                pred_bond = output_dict['bond'].cpu()
                arg_list = [(element[j], src_mask[j], pred_bond[j], pred_aroma[j], pred_charge[j], None) for j in range(b)]
                result = pool.map(result2mol, arg_list, chunksize= 64)
                result = list(result)
                pred_smiles = [item[1].split(".") for item in result] #  _, tgt_s, tgt_valid 
                pred_valid = [item[2] for item in result] #  _, tgt_s, tgt_valid 
                for j in range(b):
                    # iterate over the batch
                    flag = True
                    for item in tgts[j]:
                        if not item in pred_smiles[j]:
                            flag = False
                    if flag:
                        true_cnt += 1
                    if pred_valid[j]:
                        pred.append(pred_smiles[j])
                        valid_cnt += 1
                    else:
                        pred.append(None)
                    
                        
        test_result['acc'] = true_cnt/cnt
        test_result['valid'] = valid_cnt/cnt
        test_result['pred'] = pred
        test_result['ckpt'] = self.args.checkpoint
        print("acc: %f, valid: %f"%(test_result['acc'], test_result['valid']))
        with open("results/"+str(temperature)+ '_' + str(self.args.seed)+'.pickle', 'wb') as file:
            pickle.dump(test_result, file)
        return test_result
 


def load_data(args, name):
    file = open('data/' + args.prefix + '_' + name  + '.pickle', 'rb')
    full_data = pickle.load(file)
    file.close()
    full_dataset = TransformerDataset(args.shuffle, full_data)

    data_loader = DataLoader(full_dataset,
                             batch_size=args.batch_size,
                             shuffle=(name == 'train'),
                             num_workers=args.num_workers, collate_fn = TransformerDataset.collate_fn)
    return data_loader


                
                
if __name__ == '__main__':
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    RDLogger.DisableLog('rdApp.info') 
    
    args = parser.parse_args()
    seed = args.seed + args.local_rank
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    torch.cuda.set_device(args.local_rank)
    
    args.save_path = args.save_path + '/' + args.name + '/'
    
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    print(args)
    dist.init_process_group("nccl", rank=args.local_rank, world_size=args.world_size)

    valid_dataloader = None
    train_dataloader = None
    test_dataloader = None
    if args.train:
        valid_dataloader = load_data(args, 'valid')
        train_dataloader = load_data(args, 'train')
    if args.test:
         test_dataloader = load_data(args, 'test')

    trainer = Trainer(train_dataloader, test_dataloader, valid_dataloader, args)
    
    if args.train:
        trainer.fit()
    elif args.test:
        for temperature in args.temperature:
            trainer.test(temperature)
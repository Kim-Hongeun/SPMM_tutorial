import os
import random
import time
import argparse
from tqdm import tqdm
from pathlib import Path
import utils
import numpy as np
import torch 
from torch.backends.cudnn import cudnn
from torch.optim import optim
from pretrain_SPMM import SPMM
from dataset import SMILESDataset
from scheduler import create_scheduler

def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    model.train()

    header = 'Train Epoch: [{}] '.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_interations = warmup_steps * step_size

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    tqdm_data_loader = tqdm(data_loader, miniters=print_freq, desc=header)
    for i, (smiles, property) in enumerate(tqdm_data_loader):
        optimizer.zero_grad()
        smiles_token = tokenizer(smiles)
        smilesIds = torch.LongTensor([smiles_token['input_ids']])
        smilesAttentionMask = torch.LongTensor([smiles_token['attention_mask']])
        
        if epoch > 0:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1., i/len(data_loader))
        
        loss_psc, loss_psm, loss_nsp, loss_npp = model(property, smilesIds, smilesAttentionMask, alpha=0)
        loss = loss_psc + loss_psm + loss_nsp + loss_npp
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()

        tqdm_data_loader.set_description(f'loss_psc={loss_psc.item():.4f}, loss_psm={loss_psm.item():.4f}, loss_npp={loss_npp.item():.4f}, loss_nsp={loss_nsp.item():.4f}')

        if epoch == 0 and 1 % step_size == 0 and i <= warmup_interations:
            scheduler.step(i // step_size)

def main(args, config):
    



if __name__ == '__main__':
    parser= argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='./Pretrain/')
    parser.add_argument('--')
    args = parser.parse_args()
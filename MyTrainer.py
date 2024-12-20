#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from torch.utils.data.distributed import DistributedSampler

from pres_gpt2 import PresGPT2
from torch.optim.lr_scheduler import CosineAnnealingLR

GRAD_ACCUM = 4

class MyTrainer:
    def __init__(
        self,
        model: PresGPT2,
        train_data: DataLoader,
        validation_data: DataLoader,
        scheduler: CosineAnnealingLR,
        distributed: bool,
        fault_tolerant: bool,
        gpu_id: int,
        save_every: int,
        snapshot_path: str
    ) -> None:
        self.gpu_id = gpu_id

        if fault_tolerant:
            print('Torch Run Fault Tolerant')
            self.gpu_id = int(os.environ["LOCAL_RANK"])

        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.validation_data = validation_data
        self.scheduler = scheduler
        self.distributed = distributed
        self.fault_tolerant = fault_tolerant
        self.epochs_run = 0

        if self.fault_tolerant and os.path.exists(snapshot_path):
            print("LOADING SNAPSHOT")
            self._load_snapshot(snapshot_path)

        if distributed:
            self.model = DDP(self.model, device_ids=[self.gpu_id])
            
        self.save_every = save_every
    
    # only gets called for fault tolerant runs
    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot['MODEL_STATE'])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch):
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        torch.save(snapshot, "snapshot.pt")
        print(f"Epoch {epoch} | Training snapshot saved at snapshot.pt")

    def _run_batch(self, source, targets, counter):
        self.scheduler.optimizer.zero_grad()
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = self.model(source, targets)
        
        if self.distributed:
            self.model.require_backward_grad_sync = (counter % GRAD_ACCUM == 0)

        loss = loss / GRAD_ACCUM

        loss.backward()

        # do not step through optimizer if NOT seen 32 yet
        if counter % GRAD_ACCUM != 0:
            return loss.item()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scheduler.optimizer.step() # after this step DDP will make sure all gradients will be synced
        self.scheduler.step()
        return loss.item()

    def _run_epoch(self, epoch):
        if self.distributed and isinstance(self.train_data.sampler, DistributedSampler):
            self.train_data.sampler.set_epoch(epoch)
        
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        
        train_loss = 0
        
        self.model.train()

        total = len(self.train_data) + (GRAD_ACCUM - (len(self.train_data)%GRAD_ACCUM))

        i = 1

        while i <= total:
            for source, targets in self.train_data:
                source = source.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                train_loss += self._run_batch(source, targets, i)
                i += 1

                if i > total:
                    break

        return train_loss / (total / GRAD_ACCUM)
    
    def _calculate_validation_loss(self, epoch):
        self.model.eval()
    
        val_loss = 0
        
        with torch.no_grad():
            if self.distributed and isinstance(self.validation_data.sampler, DistributedSampler):
                self.validation_data.sampler.set_epoch(epoch)
            
            for X, Y in self.validation_data:
                X, Y = X.to(self.gpu_id), Y.to(self.gpu_id)
                _, loss = self.model(X, Y)
                
                val_loss += loss.item()
            
        val_loss_tensor = torch.tensor(val_loss, device=self.gpu_id)
        
        if self.distributed:
            torch.distributed.all_reduce(val_loss_tensor, op=torch.distributed.ReduceOp.AVG)

        return val_loss_tensor.item() / len(self.validation_data)

    def _save_checkpoint(self, epoch):
        ckp = None

        if self.distributed:
            ckp = self.model.module.state_dict()
        else:
            ckp = self.model.state_dict()
        PATH = "./model/checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")


    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            epoch_loss = self._run_epoch(epoch)
            val_loss = self._calculate_validation_loss(epoch)
            
            if (not self.distributed or self.gpu_id == 0): # only have gpu_id of 0 print stuff
                print('Train Loss: ', epoch_loss)
                print('Validation Loss: ', val_loss)
                
                if self.fault_tolerant:
                    self._save_snapshot(epoch)
                else:
                    self._save_checkpoint(epoch)

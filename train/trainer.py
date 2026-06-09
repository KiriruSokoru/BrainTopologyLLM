import torch
import random
from tqdm import tqdm
from train.checkpoint_manager import CheckpointManager
from train.trickster import active_trickster

class GPT2Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_mgr = CheckpointManager()
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Плут может перемешать токены
            if active_trickster.enabled and random.random() < 0.1:
                perm = torch.randperm(inputs.size(0))
                inputs = inputs[perm]
                targets = targets[perm]
            
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            outputs = self.model(inputs, labels=targets)
            loss = outputs.loss
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # Плут может переставить градиенты
            active_trickster.play_trick(self.model, self.optimizer)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validating"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs, labels=targets)
                total_loss += outputs.loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs, damage_info=None):
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            if active_trickster.tricks_done > 0:
                print(f"  🎭 Плут сыграл {active_trickster.tricks_done} пакостей")
            
            self.checkpoint_mgr.save(
                self.model, self.optimizer, epoch, val_loss,
                {'damage': damage_info, 'history': history, 'tricks': active_trickster.tricks_done}
            )
        
        return history

def train_model(model, train_loader, val_loader, config, device, damage_info=None):
    trainer = GPT2Trainer(model, train_loader, val_loader, config, device)
    history = trainer.train(config['training']['epochs'], damage_info)
    return history['val_loss'][-1]

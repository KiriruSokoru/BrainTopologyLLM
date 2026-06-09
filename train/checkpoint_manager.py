import os
import torch
from datetime import datetime

class CheckpointManager:
    def __init__(self, save_dir='./checkpoints'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save(self, model, optimizer, epoch, loss, metadata=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gpt2_epoch{epoch}_{timestamp}.pt"
        path = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss': loss,
            'metadata': metadata or {}
        }
        torch.save(checkpoint, path)
        print(f"💾 Чекпоинт сохранён: {path}")
        return path
    
    def load_latest(self, model, optimizer=None):
        checkpoints = [f for f in os.listdir(self.save_dir) if f.startswith('gpt2_') and f.endswith('.pt')]
        if not checkpoints:
            return 0
        latest = max(checkpoints, key=lambda f: os.path.getctime(os.path.join(self.save_dir, f)))
        path = os.path.join(self.save_dir, latest)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        print(f"📂 Загружен чекпоинт: {path} (epoch {checkpoint['epoch']})")
        return checkpoint['epoch'] + 1

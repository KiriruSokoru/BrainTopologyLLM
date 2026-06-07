import torch
import os
import json
from datetime import datetime

class CheckpointManager:
    
    def __init__(self, exp_name, checkpoint_dir='./checkpoints'):
        self.exp_name = exp_name
        self.checkpoint_dir = os.path.join(checkpoint_dir, exp_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.metadata_file = os.path.join(self.checkpoint_dir, 'metadata.json')
        self.model_file = os.path.join(self.checkpoint_dir, 'model.pt')
        self.optimizer_file = os.path.join(self.checkpoint_dir, 'optimizer.pt')
        self.history_file = os.path.join(self.checkpoint_dir, 'history.pt')
        self.surgery_log_file = os.path.join(self.checkpoint_dir, 'surgery_log.json')
    
    def save(self, model, optimizer, history, surgery_log=None, epoch=None, phase=None):
        metadata = {
            'exp_name': self.exp_name,
            'epoch': epoch,
            'phase': phase,
            'timestamp': datetime.now().isoformat(),
            'history_length': len(history.get('epoch', [])),
            'surgery_performed': surgery_log is not None and len(surgery_log) > 0
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        torch.save(model.state_dict(), self.model_file)
        torch.save(optimizer.state_dict(), self.optimizer_file)
        torch.save(history, self.history_file)
        
        if surgery_log:
            with open(self.surgery_log_file, 'w') as f:
                json.dump(surgery_log, f, indent=2)
        
        print(f"  💾 Чекпоинт сохранён: эпоха {epoch}, фаза '{phase}'")
        return True
    
    def load(self, model, optimizer):
        if not os.path.exists(self.metadata_file):
            print("  📭 Чекпоинтов не найдено, начинаем с нуля")
            return None
        
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"  📀 Найден чекпоинт: эпоха {metadata['epoch']}, фаза '{metadata['phase']}'")
        
        # Получаем device из модели
        device = next(model.parameters()).device
        
        if os.path.exists(self.model_file):
            model.load_state_dict(torch.load(self.model_file, map_location=device, weights_only=False))
            print(f"     Модель загружена")
        
        if os.path.exists(self.optimizer_file):
            optimizer.load_state_dict(torch.load(self.optimizer_file, map_location=device, weights_only=False))
            print(f"     Оптимизатор загружен")
        
        history = None
        if os.path.exists(self.history_file):
            history = torch.load(self.history_file, map_location='cpu', weights_only=False)
            print(f"     История загружена ({len(history.get('epoch', []))} эпох)")
        
        surgery_log = None
        if os.path.exists(self.surgery_log_file):
            with open(self.surgery_log_file, 'r') as f:
                surgery_log = json.load(f)
            print(f"     Лог хирургии загружен")
        
        return (metadata, history, surgery_log, model, optimizer)
    
    def clear(self):
        import shutil
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
            print(f"  🧹 Чекпоинты {self.exp_name} удалены")
    
    def list_checkpoints(self):
        if not os.path.exists(self.metadata_file):
            print("  Нет сохранённых чекпоинтов")
            return None
        
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"\n  📋 Чекпоинт {self.exp_name}:")
        print(f"     Эпоха: {metadata['epoch']}")
        print(f"     Фаза: {metadata['phase']}")
        print(f"     Время: {metadata['timestamp']}")
        print(f"     История: {metadata['history_length']} записей")
        
        return metadata

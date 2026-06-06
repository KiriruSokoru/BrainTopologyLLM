"""
Checkpoint Manager для BrainTopologyLLM
Автоматическое сохранение и восстановление экспериментов
"""

import os
import warnings
import torch
import json
from datetime import datetime

class CheckpointManager:
    """Умное сохранение/восстановление экспериментов"""
    
    def __init__(self, experiment_name, base_dir='checkpoints'):
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.metadata = {
            'start_time': datetime.now().isoformat(),
            'experiment': experiment_name,
            'resumed': False
        }
        os.makedirs(base_dir, exist_ok=True)
    
    def save(self, stage, model, optimizer=None, epoch=None, extra=None, metrics=None):
        """Сохраняет чекпоинт"""
        checkpoint = {
            'stage': stage,
            'timestamp': datetime.now().isoformat(),
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict() if optimizer else None,
            'epoch': epoch,
            'extra': extra or {},
            'metrics': metrics or {},
            'metadata': self.metadata
        }
        
        filename = f"{self.experiment_name}_{stage}.pt"
        filepath = os.path.join(self.base_dir, filename)
        torch.save(checkpoint, filepath)
        
        # Сохраняем индекс
        self._update_index(stage, filepath, metrics)
        
        print(f"💾 Чекпоинт сохранён: {stage}")
        return filepath
    
    def load(self, stage):
        """Загружает чекпоинт"""
        filename = f"{self.experiment_name}_{stage}.pt"
        filepath = os.path.join(self.base_dir, filename)
        
        if not os.path.exists(filepath):
            return None
        
        warnings.warn("Загрузка с weights_only=False — только для доверенных чекпоинтов")
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        self.metadata = checkpoint['metadata']
        self.metadata['resumed'] = True
        self.metadata['resume_time'] = datetime.now().isoformat()
        
        print(f"📂 Загружен чекпоинт: {stage} (от {checkpoint['timestamp'][:19]})")
        if checkpoint.get('metrics'):
            print(f"   Метрики: {checkpoint['metrics']}")
        return checkpoint
    
    def stage_exists(self, stage):
        """Проверяет существование чекпоинта"""
        filename = f"{self.experiment_name}_{stage}.pt"
        filepath = os.path.join(self.base_dir, filename)
        return os.path.exists(filepath)
    
    def get_latest_stage(self):
        """Возвращает самый поздний существующий этап"""
        stages = [
            'warmup_epoch10',
            'diagnosis', 
            'surgery',
            'relax_epoch25',
            'relax_epoch50',
            'final'
        ]
        for stage in reversed(stages):
            if self.stage_exists(stage):
                return stage
        return None
    
    def _update_index(self, stage, filepath, metrics=None):
        """Обновляет индекс всех чекпоинтов"""
        index_file = os.path.join(self.base_dir, f"{self.experiment_name}_index.json")
        
        index = {}
        if os.path.exists(index_file):
            with open(index_file, 'r') as f:
                index = json.load(f)
        
        index[stage] = {
            'filepath': filepath,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)
    
    def print_status(self):
        """Печатает статус всех чекпоинтов"""
        print(f"\n📋 Статус эксперимента '{self.experiment_name}':")
        stages = ['warmup_epoch10', 'diagnosis', 'surgery', 'relax_epoch25', 'relax_epoch50', 'final']
        
        for stage in stages:
            exists = "✅" if self.stage_exists(stage) else "❌"
            print(f"   {exists} {stage}")
        
        latest = self.get_latest_stage()
        if latest:
            print(f"\n▶️ Можно продолжить с этапа: {latest}")
            return latest
        return None
    
    def delete_checkpoint(self, stage):
        """Удаляет чекпоинт (для отладки)"""
        filename = f"{self.experiment_name}_{stage}.pt"
        filepath = os.path.join(self.base_dir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"🗑️ Удалён чекпоинт: {stage}")

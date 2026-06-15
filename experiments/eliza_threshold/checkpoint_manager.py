"""
Checkpoint Manager для ELIZA эксперимента
Сохраняет прогресс после каждого блока (фраза × повтор)
Позволяет продолжить с места остановки
"""
import json
import os
import pickle
from datetime import datetime
from typing import Dict, Any, Optional

class ELIZACheckpointManager:
    def __init__(self, experiment_name: str, checkpoint_dir: str = "experiments/eliza_threshold/checkpoints"):
        self.experiment_name = experiment_name
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def get_checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, f"{self.experiment_name}_checkpoint.pkl")
    
    def save_checkpoint(self, completed_combinations: list, last_index: int, 
                        accumulated_results: list, mode: str):
        """Сохраняет чекпоинт"""
        checkpoint = {
            'completed_combinations': completed_combinations,
            'last_index': last_index,
            'accumulated_results': accumulated_results,
            'mode': mode,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.get_checkpoint_path(), 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"  💾 Чекпоинт сохранён: {len(accumulated_results)} диалогов обработано")
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Загружает чекпоинт, если есть"""
        path = self.get_checkpoint_path()
        if os.path.exists(path):
            with open(path, 'rb') as f:
                checkpoint = pickle.load(f)
            print(f"  🔄 Загружен чекпоинт от {checkpoint['timestamp']}")
            print(f"     Обработано: {len(checkpoint['accumulated_results'])} диалогов")
            return checkpoint
        return None
    
    def clear_checkpoint(self):
        """Удаляет чекпоинт (при force_restart или успешном завершении)"""
        path = self.get_checkpoint_path()
        if os.path.exists(path):
            os.remove(path)
            print("  🧹 Чекпоинт удалён")

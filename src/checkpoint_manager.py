#!/usr/bin/env python3
"""
Checkpoint Manager для BrainTopologyLLM.

Автоматическое сохранение и восстановление экспериментов на разных стадиях.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

import torch

logger = logging.getLogger(__name__)

__all__ = ["CheckpointManager"]


class CheckpointManager:
    """Умное сохранение/восстановление экспериментов с поддержкой стадий."""

    def __init__(self, experiment_name: str, base_dir: str = 'checkpoints') -> None:
        """Инициализирует менеджер чекпоинтов.
        
        Args:
            experiment_name: Имя эксперимента (используется как префикс файлов).
            base_dir: Базовая директория для хранения чекпоинтов.
        """
        self.experiment_name: str = experiment_name
        self.base_dir: str = base_dir
        self.metadata: Dict[str, Any] = {
            'start_time': datetime.now().isoformat(),
            'experiment': experiment_name,
            'resumed': False
        }
        os.makedirs(base_dir, exist_ok=True)

    def save(
        self,
        stage: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """Сохраняет чекпоинт модели на указанной стадии.
        
        Args:
            stage: Название стадии эксперимента (например, 'warmup_epoch10').
            model: Модель PyTorch для сохранения.
            optimizer: Оптимизатор (опционально).
            epoch: Номер эпохи (опционально).
            extra: Дополнительные данные для сохранения (опционально).
            metrics: Метрики для сохранения (опционально).
        
        Returns:
            Путь к сохранённому файлу чекпоинта.
        """
        checkpoint: Dict[str, Any] = {
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

        logger.info(f"💾 Чекпоинт сохранён: {stage}")
        return filepath

    def load(self, stage: str) -> Optional[Dict[str, Any]]:
        """Загружает чекпоинт модели с указанной стадии.
        
        Args:
            stage: Название стадии для загрузки.
        
        Returns:
            Словарь с данными чекпоинта или None, если файл не найден.
        
        Note:
            Загрузка выполняется с weights_only=False, что требует доверенных чекпоинтов.
            Не загружайте чекпоинты из ненадёжных источников.
        """
        filename = f"{self.experiment_name}_{stage}.pt"
        filepath = os.path.join(self.base_dir, filename)

        if not os.path.exists(filepath):
            return None

        # ВНИМАНИЕ: weights_only=False требует доверенных чекпоинтов!
        # Никогда не загружайте чекпоинты из ненадёжных источников.
        logger.warning(
            "Загрузка с weights_only=False — только для доверенных чекпоинтов. "
            "Не загружайте файлы из ненадёжных источников!"
        )
        
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        # Explicitly cast to expected dict type for static checkers
        checkpoint = cast(Dict[str, Any], checkpoint)
        self.metadata = checkpoint['metadata']
        self.metadata['resumed'] = True
        self.metadata['resume_time'] = datetime.now().isoformat()

        logger.info(f"📂 Загружен чекпоинт: {stage} (от {checkpoint['timestamp'][:19]})")
        if checkpoint.get('metrics'):
            logger.info(f"   Метрики: {checkpoint['metrics']}")
        return checkpoint

    def stage_exists(self, stage: str) -> bool:
        """Проверяет существование чекпоинта для указанной стадии.
        
        Args:
            stage: Название стадии для проверки.
        
        Returns:
            True, если чекпоинт существует, иначе False.
        """
        filename = f"{self.experiment_name}_{stage}.pt"
        filepath = os.path.join(self.base_dir, filename)
        return os.path.exists(filepath)

    def get_latest_stage(self) -> Optional[str]:
        """Возвращает самый поздний существующий этап эксперимента.
        
        Returns:
            Название последней стадии или None, если чекпоинтов нет.
        """
        stages: List[str] = [
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

    def _update_index(
        self,
        stage: str,
        filepath: str,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Обновляет индекс всех чекпоинтов в JSON файле.
        
        Args:
            stage: Название стадии.
            filepath: Путь к файлу чекпоинта.
            metrics: Метрики для сохранения в индексе (опционально).
        """
        index_file = os.path.join(self.base_dir, f"{self.experiment_name}_index.json")

        index: Dict[str, Any] = {}
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

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """Возвращает список всех чекпоинтов с метаданными.
        
        Returns:
            Список словарей с информацией о каждом чекпоинте:
            [{'stage': str, 'timestamp': str, 'metrics': dict, 'filepath': str}, ...]
        """
        index_file = os.path.join(self.base_dir, f"{self.experiment_name}_index.json")
        
        if not os.path.exists(index_file):
            return []
        
        with open(index_file, 'r') as f:
            index = json.load(f)
        
        checkpoints: List[Dict[str, Any]] = []
        for stage, info in index.items():
            checkpoints.append({
                'stage': stage,
                'timestamp': info.get('timestamp', ''),
                'metrics': info.get('metrics', {}),
                'filepath': info.get('filepath', '')
            })
        
        return checkpoints

    def get_checkpoint_info(self, stage: str) -> Optional[Dict[str, Any]]:
        """Возвращает информацию о чекпоинте без загрузки тяжёлого .pt файла.
        
        Args:
            stage: Название стадии.
        
        Returns:
            Словарь с timestamp, metrics, filepath или None, если чекпоинт не найден.
        """
        index_file = os.path.join(self.base_dir, f"{self.experiment_name}_index.json")
        
        if not os.path.exists(index_file):
            return None
        
        with open(index_file, 'r') as f:
            index = json.load(f)
        
        if stage not in index:
            return None
        
        info = index[stage]
        return {
            'timestamp': info.get('timestamp', ''),
            'metrics': info.get('metrics', {}),
            'filepath': info.get('filepath', '')
        }

    def print_status(self) -> Optional[str]:
        """Печатает статус всех чекпоинтов эксперимента.
        
        Returns:
            Название последней стадии или None.
        """
        logger.info(f"\n📋 Статус эксперимента '{self.experiment_name}':")
        stages: List[str] = [
            'warmup_epoch10',
            'diagnosis',
            'surgery',
            'relax_epoch25',
            'relax_epoch50',
            'final'
        ]

        for stage in stages:
            exists = "✅" if self.stage_exists(stage) else "❌"
            logger.info(f"   {exists} {stage}")

        latest = self.get_latest_stage()
        if latest:
            logger.info(f"\n▶️ Можно продолжить с этапа: {latest}")
            return latest
        return None

    def delete_checkpoint(self, stage: str) -> None:
        """Удаляет чекпоинт (для отладки).
        
        Args:
            stage: Название стадии для удаления.
        """
        filename = f"{self.experiment_name}_{stage}.pt"
        filepath = os.path.join(self.base_dir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"🗑️ Удалён чекпоинт: {stage}")

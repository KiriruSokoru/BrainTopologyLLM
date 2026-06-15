#!/usr/bin/env python3
"""
Proof of Concept: Neural Seed (Нейронное Семя).

Демонстрирует возможность извлечения "мастеров" (топологических якорей) 
из обученной модели, архивации этого семени и "выращивания" новой модели 
с нуля за значительно меньшее количество эпох.
"""

import argparse
import logging
import os
import warnings
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.core.models import SimpleMLP
from src.core.metrics import train_model, accuracy

__all__ = ["main", "get_masters", "archive_seed", "grow_model"]

logger = logging.getLogger(__name__)


def get_masters(model: SimpleMLP, top_k: int) -> Dict[str, torch.Tensor]:
    """Извлекает веса 'мастеров' (наиболее важных нейронов) из модели.

    В данном PoC важность определяется L2-нормой весов первого слоя.
    
    Args:
        model: Обученная модель SimpleMLP.
        top_k: Количество нейронов-мастеров для извлечения.

    Returns:
        Словарь с маской мастеров, их весами и смещениями.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        weights = model.fc1.weight.data
        # Вычисляем L2-норму для каждого нейрона (строки матрицы весов)
        norms = torch.norm(weights, dim=1)
        
        # Находим индексы top_k нейронов с наибольшей нормой
        _, master_indices = torch.topk(norms, k=top_k)
        
        # Создаем маску
        mask = torch.zeros_like(norms)
        mask[master_indices] = 1.0
        
        return {
            "master_indices": master_indices,
            "mask": mask,
            "fc1_weights": weights[master_indices].clone(),
            "fc1_bias": model.fc1.bias.data[master_indices].clone(),
        }


def archive_seed(seed_data: Dict[str, torch.Tensor], output_path: str) -> None:
    """Сохраняет извлеченное семя в файл.

    Args:
        seed_data: Словарь с данными семени (веса, маски, индексы).
        output_path: Путь для сохранения .pt файла.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(seed_data, output_path)
    logger.info(f"Семя архивировано: {output_path} (размер: {os.path.getsize(output_path) / 1024:.2f} KB)")


def grow_model(
    seed_path: str, 
    hidden_size: int, 
    epochs: int, 
    device: torch.device,
    train_loader: DataLoader,
    test_loader: DataLoader
) -> float:
    """Выращивает новую модель из архивного семени.

    Args:
        seed_path: Путь к файлу семени.
        hidden_size: Размер скрытого слоя новой модели.
        epochs: Количество эпох для дообучения (выращивания).
        device: Устройство вычислений.
        train_loader: DataLoader для обучения.
        test_loader: DataLoader для валидации.

    Returns:
        Точность выращенной модели на тестовой выборке.
    """
    logger.info(f"Загрузка семени из {seed_path}...")
    seed_data = torch.load(seed_path, map_location=device, weights_only=False)
    
    # Создаем новую модель
    new_model = SimpleMLP(hidden_size=hidden_size).to(device)
    
    # Инициализируем веса мастеров из семени
    with torch.no_grad():
        master_indices = seed_data["master_indices"]
        new_model.fc1.weight.data[master_indices] = seed_data["fc1_weights"]
        new_model.fc1.bias.data[master_indices] = seed_data["fc1_bias"]
        
        # Остальные нейроны (не-мастера) остаются со случайной инициализацией 
        # или могут быть занулены в зависимости от стратегии. Здесь оставляем random.
        
    logger.info(f"Выращивание модели в течение {epochs} эпох...")
    train_model(new_model, train_loader, epochs=epochs, device=device, lr=0.005)
    
    test_acc = accuracy(new_model, test_loader, device)
    return test_acc


def main(
    hidden_size: int,
    epochs_train: int,
    epochs_grow: int,
    top_k: int,
    output_path: str
) -> None:
    """Оркестрация эксперимента Neural Seed.

    Args:
        hidden_size: Размер скрытого слоя MLP.
        epochs_train: Количество эпох для обучения эталонной модели.
        epochs_grow: Количество эпох для выращивания модели из семени.
        top_k: Количество извлекаемых нейронов-мастеров.
        output_path: Путь для сохранения файла семени.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Используемое устройство: {device}")

    # 1. Подготовка данных
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 2. Обучение эталонной модели
    logger.info(f"Обучение эталонной модели ({epochs_train} эпох)...")
    baseline_model = SimpleMLP(hidden_size=hidden_size).to(device)
    baseline_acc = train_model(baseline_model, train_loader, epochs=epochs_train, device=device)
    
    # Точная оценка на тесте
    baseline_test_acc = accuracy(baseline_model, test_loader, device)
    logger.info(f"Эталонная модель: Accuracy на тесте = {baseline_test_acc:.2%}")

    # 3. Извлечение мастеров (Семя)
    logger.info(f"Извлечение топ-{top_k} мастеров...")
    seed_data = get_masters(baseline_model, top_k=top_k)
    archive_seed(seed_data, output_path)

    # 4. Выращивание новой модели из семени
    logger.info("Начало эксперимента по выращиванию...")
    grown_test_acc = grow_model(
        seed_path=output_path,
        hidden_size=hidden_size,
        epochs=epochs_grow,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader
    )
    
    # 5. Сравнение результатов
    logger.info("=" * 60)
    logger.info("ИТОГИ ЭКСПЕРИМЕНТА NEURAL SEED")
    logger.info("=" * 60)
    logger.info(f"Эталон (обучение {epochs_train} эпох): {baseline_test_acc:.2%}")
    logger.info(f"Выращенная (обучение {epochs_grow} эпох): {grown_test_acc:.2%}")
    
    diff = grown_test_acc - baseline_test_acc
    if diff >= -0.01:  # Допуск 1%
        logger.info("✅ УСПЕХ: Выращенная модель достигла качества, близкого к эталону, за меньшее время!")
    else:
        logger.warning(f"⚠️ Отставание выращенной модели: {diff:.2%}")
    logger.info("=" * 60)


if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    
    parser = argparse.ArgumentParser(description="PoC: Извлечение и выращивание нейронного семени")
    parser.add_argument(
        '--hidden-size', type=int, default=128,
        help='Размер скрытого слоя MLP (по умолчанию: 128)'
    )
    parser.add_argument(
        '--epochs-train', type=int, default=8,
        help='Количество эпох для обучения эталонной модели (по умолчанию: 8)'
    )
    parser.add_argument(
        '--epochs-grow', type=int, default=3,
        help='Количество эпох для выращивания модели из семени (по умолчанию: 3)'
    )
    parser.add_argument(
        '--top-k', type=int, default=30,
        help='Количество извлекаемых нейронов-мастеров (по умолчанию: 30)'
    )
    parser.add_argument(
        '--output', type=str, default='results/neural_seed.pt',
        help='Путь для сохранения файла семени (по умолчанию: results/neural_seed.pt)'
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main(
        hidden_size=args.hidden_size,
        epochs_train=args.epochs_train,
        epochs_grow=args.epochs_grow,
        top_k=args.top_k,
        output_path=args.output
    )

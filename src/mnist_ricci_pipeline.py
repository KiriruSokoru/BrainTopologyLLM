#!/usr/bin/env python3
"""
MNIST Ricci Pipeline — полный пайплайн анализа топологии нейросети.

1. Загружает или обучает модель
2. Собирает активации
3. Строит граф корреляций
4. Вычисляет кривизну Риччи
5. Применяет pruning
6. Визуализирует результаты
"""

import argparse
import logging
import os
from typing import List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.core.models import MNIST_CNN
from src.core.metrics import accuracy
from src.core.activations import collect_activations
from src.core.graph_builder import build_activation_graph
from src.core.ricci import compute_ricci_curvature

logger = logging.getLogger(__name__)

torch.set_num_threads(4)


def load_or_train_model(
    checkpoint_path: str,
    epochs: int,
    device: torch.device
) -> MNIST_CNN:
    """Загружает модель из чекпоинта или обучает с нуля.
    
    Args:
        checkpoint_path: Путь к файлу чекпоинта.
        epochs: Количество эпох для обучения.
        device: Устройство вычислений.
    
    Returns:
        Обученная модель.
    """
    model = MNIST_CNN().to(device)
    
    if os.path.exists(checkpoint_path):
        logger.info(f"Загрузка модели из {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        return model
    
    logger.info(f"Чекпоинт не найден. Обучение модели на {epochs} эпох...")
    
    # Данные
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Обучение
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_acc = 100.0 * correct / total
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f} - Acc: {train_acc:.2f}%")
    
    # Сохраняем чекпоинт
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Модель сохранена в {checkpoint_path}")
    
    return model


def prune_model(
    model: MNIST_CNN,
    prune_indices: set,
    neuron_labels: List[str],
    device: torch.device
) -> MNIST_CNN:
    """Применяет маску pruning к модели.
    
    Args:
        model: Исходная модель.
        prune_indices: Индексы нейронов для удаления.
        neuron_labels: Список меток нейронов.
        device: Устройство вычислений.
    
    Returns:
        Новая модель с применённым pruning.
    """
    pruned = MNIST_CNN().to(device)
    pruned.load_state_dict(model.state_dict())
    
    with torch.no_grad():
        for idx in prune_indices:
            label = neuron_labels[idx]
            layer, neuron = label.split(':')
            neuron = int(neuron)
            
            if layer == 'conv1':
                pruned.conv1.weight[neuron] = 0.0
                pruned.conv1.bias[neuron] = 0.0
            elif layer == 'conv2':
                pruned.conv2.weight[neuron] = 0.0
                pruned.conv2.bias[neuron] = 0.0
            elif layer == 'fc1':
                pruned.fc1.weight[neuron] = 0.0
                pruned.fc1.bias[neuron] = 0.0
    
    return pruned


def main(
    epochs: int,
    prune_fractions: List[float],
    output_name: str
) -> None:
    """Оркестрация полного пайплайна анализа топологии MNIST CNN.
    
    Args:
        epochs: Количество эпох для обучения (если модель не найдена).
        prune_fractions: Список долей нейронов для pruning.
        output_name: Имя для выходных файлов.
    """
    device = torch.device('cpu')
    
    logger.info("=" * 60)
    logger.info("MNIST RICCI PIPELINE")
    logger.info("Анализ топологии нейросети через кривизну Риччи")
    logger.info("=" * 60)
    
    # 1. Загрузка/обучение модели
    checkpoint_path = 'simple_cnn_mnist.pth'
    model = load_or_train_model(checkpoint_path, epochs, device)
    model.eval()
    
    # Данные для тестирования
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    baseline = accuracy(model, test_loader, device)
    logger.info(f"Baseline accuracy: {baseline:.2f}%")
    
    # 2. Сбор активаций
    logger.info("\nСбор активаций...")
    activations = collect_activations(model, train_loader, device, max_batches=50)
    
    # 3. Построение графа
    logger.info("Построение графа корреляций...")
    G, corr_matrix, neuron_labels = build_activation_graph(activations, corr_threshold=0.7)
    n = len(neuron_labels)
    logger.info(f"Граф: {n} узлов, {G.number_of_edges()} рёбер")
    
    # 4. Вычисление кривизны
    logger.info("Вычисление кривизны Оливье-Риччи...")
    ricci = compute_ricci_curvature(G, alpha=0.5)
    logger.info(f"Ricci: min={ricci.min():.4f}, max={ricci.max():.4f}, mean={ricci.mean():.4f}")
    
    # Классификация
    mountains = np.where(ricci > 0.1)[0]
    plateaus = np.where((ricci > -0.1) & (ricci < 0.1))[0]
    singularities = np.where(ricci < -0.1)[0]
    
    logger.info(f"\nЛандшафт:")
    logger.info(f" Горы (Ricci > 0.1): {len(mountains)}")
    logger.info(f" Плато (-0.1..0.1): {len(plateaus)}")
    logger.info(f" Сингулярности (< -0.1): {len(singularities)}")
    
    # Сортировка по важности
    deg = np.array([G.degree(i) for i in range(n)])
    importance = np.abs(ricci) * (deg / (n - 1) * 5 + 1)
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-10)
    sorted_by_importance = np.argsort(importance)
    
    # 5. Pruning эксперименты
    logger.info("\n" + "=" * 60)
    logger.info("PRUNING ЭКСПЕРИМЕНТЫ")
    logger.info("=" * 60)
    
    results: List[float] = []
    
    for frac in prune_fractions:
        n_prune = int(n * frac)
        prune_set = set(sorted_by_importance[:n_prune])
        
        pruned = prune_model(model, prune_set, neuron_labels, device)
        acc = accuracy(pruned, test_loader, device)
        results.append(acc)
        
        logger.info(f"Prune {frac:.0%}: {acc:.2f}% (Δ {acc - baseline:+.2f}%)")
        del pruned
    
    # 6. Визуализация
    logger.info("\nПостроение графиков...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # График 1: Pruning curve
    axes[0].plot([0] + prune_fractions, [baseline] + results, 'bo-', linewidth=2, markersize=8)
    axes[0].axhline(y=baseline, color='red', linestyle='--', alpha=0.5, label='Baseline')
    axes[0].set_xlabel('Prune Fraction')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Accuracy vs Prune Fraction')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # График 2: Ricci distribution
    axes[1].hist(ricci, bins=40, color='steelblue', edgecolor='white', alpha=0.7)
    axes[1].axvline(x=0, color='black', linestyle='-', linewidth=2)
    axes[1].axvline(x=-0.1, color='red', linestyle='--', alpha=0.7, label='Singularities')
    axes[1].axvline(x=0.1, color='gold', linestyle='--', alpha=0.7, label='Mountains')
    axes[1].set_xlabel('Ricci Curvature')
    axes[1].set_ylabel('Neurons')
    axes[1].set_title('Ricci Curvature Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # График 3: Importance
    axes[2].plot(importance[sorted_by_importance], 'g-', linewidth=2)
    axes[2].set_xlabel('Neuron Rank')
    axes[2].set_ylabel('Importance')
    axes[2].set_title('Neuron Importance')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f'{output_name}.png'
    plt.savefig(plot_path, dpi=150)
    logger.info(f"Сохранено: {plot_path}")
    
    # 7. Сохранение данных
    npz_path = f'{output_name}.npz'
    np.savez(
        npz_path,
        baseline=baseline,
        fractions=prune_fractions,
        accuracies=results,
        ricci=ricci,
        mountains=mountains,
        singularities=singularities,
        plateaus=plateaus
    )
    logger.info(f"Данные сохранены: {npz_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("ГОТОВО!")
    logger.info("=" * 60)


if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    
    parser = argparse.ArgumentParser(description='MNIST Ricci Pipeline')
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Количество эпох для обучения (по умолчанию: 3)'
    )
    parser.add_argument(
        '--prune-fractions',
        type=float,
        nargs='+',
        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        help='Доли нейронов для pruning (по умолчанию: 0.05 0.10 0.15 0.20 0.25 0.30)'
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default='mnist_ricci_results',
        help='Имя для выходных файлов (по умолчанию: mnist_ricci_results)'
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main(
        epochs=args.epochs,
        prune_fractions=args.prune_fractions,
        output_name=args.output_name
    )

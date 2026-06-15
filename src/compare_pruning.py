#!/usr/bin/env python3
"""
Сравнение методов pruning: Ricci, Weight (Magnitude), Random.

Оценивает деградацию точности при удалении нейронов разными стратегиями.
"""

import argparse
import logging
import os
from typing import Dict, List, Set, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.core.models import MNIST_CNN
from src.core.metrics import accuracy
from src.core.activations import collect_activations
from src.core.graph_builder import build_activation_graph, parse_neuron_labels
from src.core.ricci import compute_ricci_curvature

logger = logging.getLogger(__name__)

torch.set_num_threads(4)


def get_neuron_weights(model: MNIST_CNN, neuron_map: Dict[int, Tuple[str, int]]) -> np.ndarray:
    """Вычисляет важность нейронов на основе L2-нормы их весов (Magnitude-based).

    Args:
        model: Обученная модель PyTorch.
        neuron_map: Словарь маппинга {глобальный_индекс: (имя_слоя, локальный_индекс)}.

    Returns:
        Массив важности (L2-норма) для каждого нейрона.
    """
    n_neurons = len(neuron_map)
    weights_norm = np.zeros(n_neurons)
    
    with torch.no_grad():
        for idx, (layer_name, local_idx) in neuron_map.items():
            if layer_name == 'conv1':
                w = model.conv1.weight[local_idx]
            elif layer_name == 'conv2':
                w = model.conv2.weight[local_idx]
            elif layer_name == 'fc1':
                w = model.fc1.weight[local_idx]
            else:
                continue
            
            # L2-норма весов нейрона как мера его важности
            weights_norm[idx] = torch.norm(w).item()
            
    return weights_norm


def prune_model(
    model: MNIST_CNN, 
    prune_indices: Set[int], 
    neuron_map: Dict[int, Tuple[str, int]], 
    device: torch.device
) -> MNIST_CNN:
    """Применяет маску pruning (зануление) к указанным нейронам.

    Args:
        model: Исходная модель.
        prune_indices: Множество глобальных индексов нейронов для удаления.
        neuron_map: Словарь маппинга индексов.
        device: Устройство вычислений.

    Returns:
        Новая модель с применённым pruning.
    """
    pruned = MNIST_CNN().to(device)
    pruned.load_state_dict(model.state_dict())
    
    with torch.no_grad():
        for idx in prune_indices:
            layer_name, local_idx = neuron_map[idx]
            
            if layer_name == 'conv1':
                pruned.conv1.weight[local_idx] = 0.0,  # type: ignore
                pruned.conv1.bias[local_idx] = 0.0,  # type: ignore
            elif layer_name == 'conv2':
                pruned.conv2.weight[local_idx] = 0.0  # type: ignore
                pruned.conv2.bias[local_idx] = 0.0  # type: ignore
            elif layer_name == 'fc1':
                pruned.fc1.weight[local_idx] = 0.0  # type: ignore
                pruned.fc1.bias[local_idx] = 0.0  # type: ignore
                
    return pruned


def main(
    methods: List[str],
    fractions: List[float],
    checkpoint_path: str,
    output_dir: str
) -> None:
    """Оркестрация эксперимента по сравнению методов pruning.

    Args:
        methods: Список методов для сравнения ('Ricci', 'Weight', 'Random').
        fractions: Список долей нейронов для удаления.
        checkpoint_path: Путь к предобученной модели.
        output_dir: Директория для сохранения результатов.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cpu')
    
    logger.info("=" * 60)
    logger.info("СРАВНЕНИЕ МЕТОДОВ PRUNING")
    logger.info(f"Методы: {', '.join(methods)}")
    logger.info(f"Доли: {fractions}")
    logger.info("=" * 60)

    # 1. Данные
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 2. Модель
    if not os.path.exists(checkpoint_path):
        logger.error(f"Чекпоинт не найден: {checkpoint_path}. Сначала запустите mnist_ricci_pipeline.py")
        return
        
    model = MNIST_CNN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    baseline = accuracy(model, test_loader, device)
    logger.info(f"Baseline accuracy: {baseline:.2f}%")

    # 3. Сбор данных для анализа топологии
    logger.info("Сбор активаций и построение графа...")
    activations = collect_activations(model, train_loader, device, max_batches=50)
    G, _, neuron_labels = build_activation_graph(activations, corr_threshold=0.7)
    n = len(neuron_labels)
    neuron_map = parse_neuron_labels(neuron_labels)
    
    logger.info("Вычисление кривизны Риччи...")
    ricci = compute_ricci_curvature(G, alpha=0.5)
    
    # 4. Расчёт важности для каждого метода
    logger.info("Расчёт метрик важности нейронов...")
    importance_scores: Dict[str, np.ndarray] = {}
    
    if 'Ricci' in methods:
        # Для Ricci важность = модуль кривизны (чем ближе к 0 или отрицательная, тем менее важен/более сингулярен)
        # Но для pruning мы удаляем наименее важные. В нашем контексте "мастера" (высокий Ricci) важны.
        # Поэтому сортируем по возрастанию Ricci (удаляем сначала сингулярности/плато).
        importance_scores['Ricci'] = ricci 
        
    if 'Weight' in methods:
        importance_scores['Weight'] = get_neuron_weights(model, neuron_map)
        
    if 'Random' in methods:
        importance_scores['Random'] = np.random.rand(n)

    # Сортировка индексов по возрастанию важности (чтобы удалять наименее важные первыми)
    sorted_indices: Dict[str, np.ndarray] = {}
    for method in methods:
        sorted_indices[method] = np.argsort(importance_scores[method])

    # 5. Эксперимент по pruning
    logger.info("\nЗапуск экспериментов по pruning...")
    results: Dict[str, List[float]] = {method: [] for method in methods}
    
    for frac in fractions:
        n_prune = int(n * frac)
        logger.info(f"--- Prune {frac:.0%} ({n_prune}/{n} нейронов) ---")
        
        for method in methods:
            prune_set = set(sorted_indices[method][:n_prune])
            pruned_model = prune_model(model, prune_set, neuron_map, device)
            acc = accuracy(pruned_model, test_loader, device)
            results[method].append(acc)
            logger.info(f"  {method:6s}: {acc:.2f}% (Δ {acc - baseline:+.2f}%)")
            del pruned_model

    # 6. Визуализация
    logger.info("\nПостроение графиков...")
    plt.figure(figsize=(10, 6))
    
    colors = {'Ricci': 'green', 'Weight': 'blue', 'Random': 'red'}
    markers = {'Ricci': 'o', 'Weight': 's', 'Random': '^'}
    
    for method in methods:
        plt.plot(
            [0.0] + fractions, 
            [baseline] + results[method], 
            color=colors.get(method, 'gray'), 
            marker=markers.get(method, 'o'),
            linewidth=2, 
            markersize=8, 
            label=f"{method} Pruning"
        )
        
    plt.axhline(y=baseline, color='black', linestyle='--', alpha=0.5, label='Baseline')
    plt.xlabel('Prune Fraction')
    plt.ylabel('Accuracy (%)')
    plt.title('Сравнение методов pruning на MNIST CNN')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.02, max(fractions) + 0.02)
    
    plot_path = os.path.join(output_dir, 'pruning_comparison.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"График сохранён: {plot_path}")

    # 7. Сохранение данных
    npz_path = os.path.join(output_dir, 'pruning_comparison.npz')
    # Явно формируем поля для сохранения (без использования **dict)
    results_ricci = results.get('Ricci', [])
    results_weight = results.get('Weight', [])
    results_random = results.get('Random', [])

    importance_ricci = importance_scores.get('Ricci', np.array([]))
    importance_weight = importance_scores.get('Weight', np.array([]))

    np.savez(
        npz_path,
        baseline=baseline,
        fractions=fractions,
        results_ricci=results_ricci,
        results_weight=results_weight,
        results_random=results_random,
        importance_ricci=importance_ricci,
        importance_weight=importance_weight,
        ricci_values=ricci,
        neuron_labels=neuron_labels,
    )
    logger.info(f"Данные сохранены: {npz_path}")
    logger.info("=" * 60)
    logger.info("ЭКСПЕРИМЕНТ ЗАВЕРШЁН УСПЕШНО")
    logger.info("=" * 60)


if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    
    parser = argparse.ArgumentParser(description='Сравнение методов pruning (Ricci, Weight, Random)')
    parser.add_argument(
        '--methods',
        nargs='+',
        choices=['Ricci', 'Weight', 'Random', 'all'],
        default=['all'],
        help='Методы для сравнения (по умолчанию: all)'
    )
    parser.add_argument(
        '--fractions',
        type=float,
        nargs='+',
        default=[0.1, 0.2, 0.3, 0.4, 0.5],
        help='Доли нейронов для удаления (по умолчанию: 0.1 0.2 0.3 0.4 0.5)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='simple_cnn_mnist.pth',
        help='Путь к предобученному чекпоинту модели (по умолчанию: simple_cnn_mnist.pth)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Директория для сохранения графиков и данных (по умолчанию: results)'
    )
    
    args = parser.parse_args()
    
    # Обработка 'all'
    if 'all' in args.methods:
        methods_to_run = ['Ricci', 'Weight', 'Random']
    else:
        methods_to_run = args.methods
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main(
        methods=methods_to_run,
        fractions=args.fractions,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir
    )

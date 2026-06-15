#!/usr/bin/env python3
"""
Хирургия Перельмана на MNIST CNN.

Вместо зануления — заменяем сингулярности на среднее соседей.
"""

import argparse
import logging
import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import SpectralEmbedding
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.core.models import MNIST_CNN
from src.core.metrics import accuracy
from src.core.activations import collect_activations
from src.core.graph_builder import build_activation_graph
from src.core.ricci import compute_ricci_curvature

logger = logging.getLogger(__name__)

torch.set_num_threads(4)


def get_layer_sizes(neuron_labels: List[str]) -> Dict[str, int]:
    """Возвращает размеры слоёв на основе меток нейронов.
    
    Args:
        neuron_labels: Список меток вида "layer_name:neuron_idx".
    
    Returns:
        Словарь {layer_name: size}.
    """
    sizes: Dict[str, int] = {}
    for label in neuron_labels:
        layer, idx = label.split(':')
        idx = int(idx)
        if layer not in sizes:
            sizes[layer] = 0
        sizes[layer] = max(sizes[layer], idx + 1)
    return sizes


def surgery_replace(
    model: torch.nn.Module,
    neuron_labels: List[str],
    adjacency: np.ndarray,
    prune_indices: Set[int],
    device: torch.device
) -> torch.nn.Module:
    """Хирургия Перельмана: заменяет нейрон на среднее соседей.
    
    Для каждого удаляемого нейрона:
    1. Находит его соседей по графу
    2. Считает средний вес соседей
    3. Заменяет удаляемый нейрон этим средним
    
    Args:
        model: Исходная модель.
        neuron_labels: Список меток нейронов.
        adjacency: Матрица смежности графа.
        prune_indices: Индексы нейронов для замены.
        device: Устройство вычислений.
    
    Returns:
        Новая модель с применённой хирургией.
    """
    layer_sizes = get_layer_sizes(neuron_labels)
    
    # Маппинг: глобальный индекс -> (слой, локальный индекс)
    neuron_map: Dict[int, Tuple[str, int]] = {}
    for idx, label in enumerate(neuron_labels):
        layer, neuron = label.split(':')
        neuron_map[idx] = (layer, int(neuron))
    
    # Клонируем модель
    new_model = MNIST_CNN().to(device)
    new_model.load_state_dict(model.state_dict())
    
    with torch.no_grad():
        for global_idx in prune_indices:
            layer_name, local_idx = neuron_map[global_idx]
            
            # Находим соседей по графу (только в том же слое)
            neighbors: List[int] = []
            for neighbor_idx in range(len(neuron_labels)):
                if neighbor_idx != global_idx and adjacency[global_idx, neighbor_idx] > 0:
                    neighbor_layer, neighbor_local = neuron_map[neighbor_idx]
                    if neighbor_layer == layer_name:
                        neighbors.append(neighbor_local)
            
            if not neighbors:
                continue  # нет соседей — пропускаем
            
            # Считаем среднее значение соседей для этого нейрона
            if layer_name == 'conv1':
                avg_weight = torch.zeros_like(new_model.conv1.weight[local_idx])
                avg_bias = torch.zeros_like(new_model.conv1.bias[local_idx])
                for n in neighbors:
                    avg_weight += new_model.conv1.weight[n]
                    avg_bias += new_model.conv1.bias[n]
                avg_weight /= len(neighbors)
                avg_bias /= len(neighbors)
                new_model.conv1.weight[local_idx] = avg_weight
                new_model.conv1.bias[local_idx] = avg_bias
            
            elif layer_name == 'conv2':
                avg_weight = torch.zeros_like(new_model.conv2.weight[local_idx])
                avg_bias = torch.zeros_like(new_model.conv2.bias[local_idx])
                for n in neighbors:
                    avg_weight += new_model.conv2.weight[n]
                    avg_bias += new_model.conv2.bias[n]
                avg_weight /= len(neighbors)
                avg_bias /= len(neighbors)
                new_model.conv2.weight[local_idx] = avg_weight
                new_model.conv2.bias[local_idx] = avg_bias
            
            elif layer_name == 'fc1':
                avg_weight = torch.zeros_like(new_model.fc1.weight[local_idx])
                avg_bias = torch.zeros_like(new_model.fc1.bias[local_idx])
                for n in neighbors:
                    avg_weight += new_model.fc1.weight[n]
                    avg_bias += new_model.fc1.bias[n]
                avg_weight /= len(neighbors)
                avg_bias /= len(neighbors)
                new_model.fc1.weight[local_idx] = avg_weight
                new_model.fc1.bias[local_idx] = avg_bias
    
    return new_model


def main(
    prune_fractions: List[float],
    corr_threshold: float,
    output_dir: str
) -> None:
    """Оркестрация эксперимента по хирургии Перельмана.
    
    Args:
        prune_fractions: Список долей нейронов для pruning.
        corr_threshold: Порог корреляции для построения графа.
        output_dir: Директория для сохранения результатов.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cpu')
    
    logger.info("=" * 60)
    logger.info("ХИРУРГИЯ ПЕРЕЛЬМАНА")
    logger.info("Замена сингулярностей на стандартные колпачки")
    logger.info("=" * 60)
    
    # Данные
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Модель
    model = MNIST_CNN().to(device)
    model.load_state_dict(torch.load('simple_cnn_mnist.pth', map_location=device))
    model.eval()
    
    baseline = accuracy(model, test_loader, device)
    logger.info(f"Baseline accuracy: {baseline:.2f}%")
    
    # Активации
    logger.info("\nCollecting activations (50 batches)...")
    all_acts = collect_activations(model, train_loader, device, max_batches=50)
    
    # Строим граф
    logger.info("Building graph...")
    G, corr_matrix, neuron_labels = build_activation_graph(all_acts, corr_threshold=corr_threshold)
    n = len(neuron_labels)
    adjacency = np.zeros((n, n))
    for i, j, data in G.edges(data=True):
        adjacency[i, j] = data.get('weight', 1.0)
        adjacency[j, i] = adjacency[i, j]
    
    logger.info(f"Graph: {n} nodes, {G.number_of_edges()} edges")
    
    # Ricci
    logger.info("\nComputing Ollivier-Ricci curvature...")
    ricci = compute_ricci_curvature(G, alpha=0.5)
    logger.info(f"Ricci: min={ricci.min():.4f}, max={ricci.max():.4f}, mean={ricci.mean():.4f}")
    
    # Классифицируем нейроны
    mountains = np.where(ricci > 0.1)[0]  # горы (положительная кривизна)
    plateaus = np.where((ricci > -0.1) & (ricci < 0.1))[0]  # плато
    singularities = np.where(ricci < -0.1)[0]  # сингулярности (отрицательная)
    
    logger.info(f"\nЛандшафт:")
    logger.info(f" Горы (Ricci > 0.1): {len(mountains)} нейронов — мастера")
    logger.info(f" Плато (-0.1..0.1): {len(plateaus)} нейронов — стажёры")
    logger.info(f" Сингулярности (< -0.1): {len(singularities)} нейронов — хаотики")
    
    # Сортируем по важности (модуль Ricci * degree)
    deg = np.array([G.degree(i) for i in range(n)])
    importance = np.abs(ricci) * (deg / (n - 1) * 5 + 1)
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-10)
    sorted_by_importance = np.argsort(importance)
    
    # ==========================================================================
    # ТРИ МЕТОДА СРАВНЕНИЯ
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("СРАВНЕНИЕ МЕТОДОВ")
    logger.info("=" * 60)
    
    results_hard: List[float] = []  # жёсткое зануление
    results_surgery: List[float] = []  # хирургия Перельмана
    results_random: List[float] = []  # случайный
    
    # Маппинг для pruning
    neuron_map: Dict[int, Tuple[str, int]] = {}
    for idx, label in enumerate(neuron_labels):
        layer, neuron = label.split(':')
        neuron_map[idx] = (layer, int(neuron))
    
    for frac in prune_fractions:
        n_prune = int(n * frac)
        logger.info(f"\n--- Prune {frac:.0%} ({n_prune}/{n}) ---")
        
        # 1. Жёсткое зануление
        prune_set = set(sorted_by_importance[:n_prune])
        pm = MNIST_CNN().to(device)
        pm.load_state_dict(model.state_dict())
        
        with torch.no_grad():
            layer_masks = defaultdict(lambda: torch.ones(200))
            for idx in prune_set:
                layer, neuron = neuron_map[idx]
                layer_masks[layer][neuron] = 0.0
            
            if 'conv1' in layer_masks:
                m = layer_masks['conv1'][:16].to(device)
                pm.conv1.weight.data *= m.view(-1, 1, 1, 1)
                pm.conv1.bias.data *= m
            
            if 'conv2' in layer_masks:
                m = layer_masks['conv2'][:32].to(device)
                pm.conv2.weight.data *= m.view(-1, 1, 1, 1)
                pm.conv2.bias.data *= m
            
            if 'fc1' in layer_masks:
                m = layer_masks['fc1'][:128].to(device)
                pm.fc1.weight.data *= m.unsqueeze(1)
                pm.fc1.bias.data *= m
        
        acc_hard = accuracy(pm, test_loader, device)
        results_hard.append(acc_hard)
        logger.info(f" Hard prune: {acc_hard:.2f}% (Δ {acc_hard - baseline:+.2f}%)")
        del pm
        
        # 2. Хирургия Перельмана
        pm_surgery = surgery_replace(model, neuron_labels, adjacency, prune_set, device)
        acc_surgery = accuracy(pm_surgery, test_loader, device)
        results_surgery.append(acc_surgery)
        logger.info(f" Perelman surgery: {acc_surgery:.2f}% (Δ {acc_surgery - baseline:+.2f}%)")
        del pm_surgery
        
        # 3. Random
        random_set = set(np.random.choice(n, n_prune, replace=False))
        pm_random = MNIST_CNN().to(device)
        pm_random.load_state_dict(model.state_dict())
        
        with torch.no_grad():
            layer_masks = defaultdict(lambda: torch.ones(200))
            for idx in random_set:
                layer, neuron = neuron_map[idx]
                layer_masks[layer][neuron] = 0.0
            
            if 'conv1' in layer_masks:
                m = layer_masks['conv1'][:16].to(device)
                pm_random.conv1.weight.data *= m.view(-1, 1, 1, 1)
                pm_random.conv1.bias.data *= m
            
            if 'conv2' in layer_masks:
                m = layer_masks['conv2'][:32].to(device)
                pm_random.conv2.weight.data *= m.view(-1, 1, 1, 1)
                pm_random.conv2.bias.data *= m
            
            if 'fc1' in layer_masks:
                m = layer_masks['fc1'][:128].to(device)
                pm_random.fc1.weight.data *= m.unsqueeze(1)
                pm_random.fc1.bias.data *= m
        
        acc_random = accuracy(pm_random, test_loader, device)
        results_random.append(acc_random)
        logger.info(f" Random prune: {acc_random:.2f}% (Δ {acc_random - baseline:+.2f}%)")
        del pm_random
    
    # ==========================================================================
    # ГРАФИКИ
    # ==========================================================================
    logger.info("\nGenerating plots...")
    
    fig = plt.figure(figsize=(18, 6))
    
    # График 1: Сравнение методов
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot([0] + prune_fractions, [baseline] + results_hard, 'bo-', linewidth=2, markersize=7, label='Hard prune')
    ax1.plot([0] + prune_fractions, [baseline] + results_surgery, 'go-', linewidth=2, markersize=7, label='Perelman surgery')
    ax1.plot([0] + prune_fractions, [baseline] + results_random, 'ro-', linewidth=2, markersize=7, label='Random')
    ax1.axhline(y=baseline, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Prune Fraction')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Perelman Surgery vs Hard Prune')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График 2: Ландшафт (3D-поверхность)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    
    # Используем спектральное вложение для координат
    if n > 10:
        embedding = SpectralEmbedding(n_components=2, affinity='precomputed')
        adj_for_embed = adjacency.copy()
        adj_for_embed[adj_for_embed > 0] = 1
        coords = embedding.fit_transform(adj_for_embed)
    else:
        coords = np.random.randn(n, 2)
    
    colors_landscape = []
    for r in ricci:
        if r > 0.1:
            colors_landscape.append('gold')  # горы — золотые
        elif r < -0.1:
            colors_landscape.append('red')  # сингулярности — красные
        else:
            colors_landscape.append('lightblue')  # плато — голубые
    
    ax2.scatter(coords[:, 0], coords[:, 1], ricci, c=colors_landscape, s=30, alpha=0.7)
    ax2.set_xlabel('Dim 1')
    ax2.set_ylabel('Dim 2')
    ax2.set_zlabel('Ricci Curvature')
    ax2.set_title('Поверхность смысла\n(золото=мастера, красный=сингулярности, синий=плато)')
    
    # График 3: Гистограмма
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.hist(ricci, bins=40, color='steelblue', edgecolor='white', alpha=0.7)
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax3.axvline(x=-0.1, color='red', linestyle='--', alpha=0.7)
    ax3.axvline(x=0.1, color='gold', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Ricci Curvature')
    ax3.set_ylabel('Neurons')
    ax3.set_title(f'Ландшафт: {len(mountains)} гор, {len(singularities)} сингулярностей, {len(plateaus)} плато')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'perelman_surgery.png')
    plt.savefig(plot_path, dpi=150)
    logger.info(f"Saved: {plot_path}")
    
    # ==========================================================================
    # ИТОГИ
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("ИТОГИ")
    logger.info("=" * 60)
    logger.info(f"Baseline: {baseline:.2f}%")
    logger.info(f"\n{'Fraction':>8s}{'Hard':>8s}{'Surgery':>8s}{'Random':>8s}")
    logger.info("-" * 40)
    
    surgery_wins = 0
    for i, frac in enumerate(prune_fractions):
        h = results_hard[i]
        s = results_surgery[i]
        r = results_random[i]
        winner = ""
        if s >= h and s >= r:
            surgery_wins += 1
            winner = "← лучший"
        logger.info(f"{frac:>7.0%}{h:>7.2f}% {s:>7.2f}% {r:>7.2f}% {winner}")
    
    logger.info(f"\nХирургия Перельмана победила в {surgery_wins}/{len(prune_fractions)} случаях")
    
    # Сохраняем данные
    npz_path = os.path.join(output_dir, 'perelman_surgery.npz')
    np.savez(
        npz_path,
        baseline=baseline,
        fractions=prune_fractions,
        hard=results_hard,
        surgery=results_surgery,
        random=results_random,
        ricci=ricci,
        mountains=mountains,
        singularities=singularities,
        plateaus=plateaus
    )
    logger.info(f"\nГотово! Открой {plot_path} — там поверхность смысла.")


if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    
    parser = argparse.ArgumentParser(description='Хирургия Перельмана на MNIST CNN')
    parser.add_argument(
        '--prune-fractions',
        type=float,
        nargs='+',
        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        help='Доли нейронов для pruning (по умолчанию: 0.05 0.10 0.15 0.20 0.25 0.30)'
    )
    parser.add_argument(
        '--corr-threshold',
        type=float,
        default=0.7,
        help='Порог корреляции для построения графа (по умолчанию: 0.7)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Директория для сохранения результатов (по умолчанию: results)'
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main(
        prune_fractions=args.prune_fractions,
        corr_threshold=args.corr_threshold,
        output_dir=args.output_dir
    )

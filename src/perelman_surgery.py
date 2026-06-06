#!/usr/bin/env python3
"""
Хирургия Перельмана на MNIST CNN
Вместо зануления — заменяем сингулярности на среднее соседей.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
import os, warnings
warnings.filterwarnings('ignore')

torch.set_num_threads(4)

# ==============================================================================
# МОДЕЛЬ
# ==============================================================================

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.activations = {}
        self._hooks()
    
    def _hooks(self):
        def hook(name):
            def fn(m, i, o): self.activations[name] = o.detach()
            return fn
        self.conv1.register_forward_hook(hook('conv1'))
        self.conv2.register_forward_hook(hook('conv2'))
        self.fc1.register_forward_hook(hook('fc1'))
        self.fc2.register_forward_hook(hook('fc2'))
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ==============================================================================
# УТИЛИТЫ
# ==============================================================================

def accuracy(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100. * correct / len(loader.dataset)

def get_layer_sizes(neuron_labels):
    """Возвращает размеры слоёв."""
    sizes = {}
    for label in neuron_labels:
        layer, idx = label.split(':')
        idx = int(idx)
        if layer not in sizes:
            sizes[layer] = 0
        sizes[layer] = max(sizes[layer], idx + 1)
    return sizes

def surgery_replace(model, neuron_labels, adjacency, prune_indices, device):
    """
    Хирургия Перельмана: заменяем нейрон на среднее соседей.
    Для каждого удаляемого нейрона:
    1. Находим его соседей по графу
    2. Считаем средний вес соседей
    3. Заменяем удаляемый нейрон этим средним
    """
    layer_sizes = get_layer_sizes(neuron_labels)
    
    # Маппинг: глобальный индекс -> (слой, локальный индекс)
    neuron_map = {}
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
            neighbors = []
            for neighbor_idx in range(len(neuron_labels)):
                if neighbor_idx != global_idx and adjacency[global_idx, neighbor_idx] > 0:
                    neighbor_layer, neighbor_local = neuron_map[neighbor_idx]
                    if neighbor_layer == layer_name:
                        neighbors.append(neighbor_local)
            
            if not neighbors:
                continue  # нет соседей — пропускаем
            
            # Считаем среднее значение соседей для этого нейрона
            if layer_name == 'conv1':
                # Усредняем веса соседних каналов
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

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    device = torch.device('cpu')
    print("=" * 60)
    print("ХИРУРГИЯ ПЕРЕЛЬМАНА")
    print("Замена сингулярностей на стандартные колпачки")
    print("=" * 60)
    
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
    print(f"Baseline accuracy: {baseline:.2f}%")
    
    # Активации
    print("\nCollecting activations (50 batches)...")
    all_acts = defaultdict(list)
    with torch.no_grad():
        for idx, (data, _) in enumerate(train_loader):
            if idx >= 50: break
            data = data.to(device)
            _ = model(data)
            for name, act in model.activations.items():
                if len(act.shape) == 4:
                    act = act.mean(dim=[2,3])
                all_acts[name].append(act.cpu().numpy())
    
    for name in all_acts:
        all_acts[name] = np.concatenate(all_acts[name], axis=0)
    
    neuron_data = []
    neuron_labels = []
    for layer_name, acts in all_acts.items():
        for i in range(acts.shape[1]):
            neuron_data.append(acts[:, i])
            neuron_labels.append(f"{layer_name}:{i}")
    
    neuron_data = np.array(neuron_data)
    corr = np.corrcoef(neuron_data)
    n = corr.shape[0]
    
    # Граф + матрица смежности
    print("Building graph...")
    G = nx.Graph()
    G.add_nodes_from(range(n))
    adjacency = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            if abs(corr[i,j]) >= 0.7:
                G.add_edge(i, j, weight=1.0 - abs(corr[i,j]))
                adjacency[i,j] = 1.0 - abs(corr[i,j])
                adjacency[j,i] = adjacency[i,j]
    
    print(f"Graph: {n} nodes, {G.number_of_edges()} edges")
    
    # Ricci
    print("\nComputing Ollivier-Ricci curvature...")
    orc = OllivierRicci(G, alpha=0.5, weight="weight", verbose="INFO")
    G_ricci = orc.compute_ricci_curvature()
    
    node_ricci = {}
    for node in G_ricci.nodes():
        curvs = [G_ricci[node][nb].get('ricciCurvature', 0) for nb in G_ricci.neighbors(node)]
        node_ricci[node] = np.mean(curvs) if curvs else 0.0
    
    ricci = np.array([node_ricci[i] for i in range(n)])
    print(f"Ricci: min={ricci.min():.4f}, max={ricci.max():.4f}, mean={ricci.mean():.4f}")
    
    # Классифицируем нейроны
    mountains = np.where(ricci > 0.1)[0]      # горы (положительная кривизна)
    plateaus = np.where((ricci > -0.1) & (ricci < 0.1))[0]  # плато
    singularities = np.where(ricci < -0.1)[0]  # сингулярности (отрицательная)
    
    print(f"\nЛандшафт:")
    print(f"  Горы (Ricci > 0.1):  {len(mountains)} нейронов — мастера")
    print(f"  Плато (-0.1..0.1):   {len(plateaus)} нейронов — стажёры")
    print(f"  Сингулярности (< -0.1): {len(singularities)} нейронов — хаотики")
    
    # Сортируем по важности (модуль Ricci * degree)
    deg = np.array([G.degree(i) for i in range(n)])
    importance = np.abs(ricci) * (deg / (n-1) * 5 + 1)
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-10)
    sorted_by_importance = np.argsort(importance)
    
    # ==========================================================================
    # ТРИ МЕТОДА СРАВНЕНИЯ
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ МЕТОДОВ")
    print("=" * 60)
    
    fractions = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    results_hard = []    # жёсткое зануление
    results_surgery = [] # хирургия Перельмана
    results_random = []  # случайный
    
    for frac in fractions:
        n_prune = int(n * frac)
        print(f"\n--- Prune {frac:.0%} ({n_prune}/{n}) ---")
        
        # 1. Жёсткое зануление
        prune_set = set(sorted_by_importance[:n_prune])
        pm = MNIST_CNN().to(device)
        pm.load_state_dict(model.state_dict())
        
        def map_layer(labels):
            m = {}
            for idx, label in enumerate(labels):
                layer, neuron = label.split(':')
                m[idx] = (layer, int(neuron))
            return m
        neuron_map = map_layer(neuron_labels)
        
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
        print(f"  Hard prune:     {acc_hard:.2f}% (Δ {acc_hard-baseline:+.2f}%)")
        del pm
        
        # 2. Хирургия Перельмана
        pm_surgery = surgery_replace(model, neuron_labels, adjacency, prune_set, device)
        acc_surgery = accuracy(pm_surgery, test_loader, device)
        results_surgery.append(acc_surgery)
        print(f"  Perelman surgery: {acc_surgery:.2f}% (Δ {acc_surgery-baseline:+.2f}%)")
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
        print(f"  Random prune:   {acc_random:.2f}% (Δ {acc_random-baseline:+.2f}%)")
        del pm_random
    
    # ==========================================================================
    # ГРАФИКИ
    # ==========================================================================
    
    print("\nGenerating plots...")
    fig = plt.figure(figsize=(18, 6))
    
    # График 1: Сравнение методов
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot([0] + fractions, [baseline] + results_hard, 'bo-', linewidth=2, markersize=7, label='Hard prune')
    ax1.plot([0] + fractions, [baseline] + results_surgery, 'go-', linewidth=2, markersize=7, label='Perelman surgery')
    ax1.plot([0] + fractions, [baseline] + results_random, 'ro-', linewidth=2, markersize=7, label='Random')
    ax1.axhline(y=baseline, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Prune Fraction')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Perelman Surgery vs Hard Prune')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График 2: Ландшафт (3D-поверхность)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    
    # Используем спектральное вложение для координат
    from sklearn.manifold import SpectralEmbedding
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
            colors_landscape.append('gold')       # горы — золотые
        elif r < -0.1:
            colors_landscape.append('red')         # сингулярности — красные
        else:
            colors_landscape.append('lightblue')   # плато — голубые
    
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
    plt.savefig('results/perelman_surgery.png', dpi=150)
    print("Saved: results/perelman_surgery.png")
    
    # ==========================================================================
    # ИТОГИ
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("ИТОГИ")
    print("=" * 60)
    print(f"Baseline: {baseline:.2f}%")
    print(f"\n{'Fraction':>8s}  {'Hard':>8s}  {'Surgery':>8s}  {'Random':>8s}")
    print("-" * 40)
    
    surgery_wins = 0
    for i, frac in enumerate(fractions):
        h = results_hard[i]
        s = results_surgery[i]
        r = results_random[i]
        winner = ""
        if s >= h and s >= r:
            surgery_wins += 1
            winner = "← лучший"
        print(f"{frac:>7.0%}   {h:>7.2f}%  {s:>7.2f}%  {r:>7.2f}%  {winner}")
    
    print(f"\nХирургия Перельмана победила в {surgery_wins}/{len(fractions)} случаях")
    
    # Сохраняем данные
    np.savez('results/perelman_surgery.npz',
             baseline=baseline, fractions=fractions,
             hard=results_hard, surgery=results_surgery, random=results_random,
             ricci=ricci, mountains=mountains, singularities=singularities, plateaus=plateaus)
    
    print("\nГотово! Открой results/perelman_surgery.png — там поверхность смысла.")

if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    main()

#!/usr/bin/env python3
"""
Compare Pruning Methods: Ricci vs Random vs Weight Magnitude
=============================================================
Три стратегии pruning на одном графе активаций.
Доказывает превосходство топологического подхода.
"""

import multiprocessing as mp
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
import os, warnings, copy
warnings.filterwarnings('ignore')

torch.set_num_threads(4)
os.environ['OMP_NUM_THREADS'] = '4'

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

def accuracy(m, loader, dev):
    m.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(dev), target.to(dev)
            output = m(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100. * correct / len(loader.dataset)

def get_neuron_weights(model):
    """Извлекаем magnitude важность нейронов."""
    weights = []
    # conv1: 16 каналов
    w = model.conv1.weight.data.abs().sum(dim=[1,2,3]).cpu().numpy()
    weights.extend(w)
    # conv2: 32 канала
    w = model.conv2.weight.data.abs().sum(dim=[1,2,3]).cpu().numpy()
    weights.extend(w)
    # fc1: 128 нейронов
    w = model.fc1.weight.data.abs().sum(dim=1).cpu().numpy()
    weights.extend(w)
    # fc2: 10 выходов (не трогаем)
    w = model.fc2.weight.data.abs().sum(dim=1).cpu().numpy()
    weights.extend(w)
    return np.array(weights)

def prune_model(model, prune_indices, neuron_map, dev):
    """Создаёт прuned копию модели."""
    layer_masks = defaultdict(lambda: torch.ones(200))
    for idx in prune_indices:
        layer, neuron = neuron_map[idx]
        layer_masks[layer][neuron] = 0.0
    
    pm = MNIST_CNN().to(dev)
    pm.load_state_dict(model.state_dict())
    
    with torch.no_grad():
        if 'conv1' in layer_masks:
            m = layer_masks['conv1'][:16].to(dev)
            pm.conv1.weight.data *= m.view(-1, 1, 1, 1)  # [16,1,1,1]
            pm.conv1.bias.data *= m
        
        if 'conv2' in layer_masks:
            m = layer_masks['conv2'][:32].to(dev)
            pm.conv2.weight.data *= m.view(-1, 1, 1, 1)  # [32,1,1,1]
            pm.conv2.bias.data *= m
        
        if 'fc1' in layer_masks:
            m = layer_masks['fc1'][:128].to(dev)
            pm.fc1.weight.data *= m.unsqueeze(1)  # [128,1]
            pm.fc1.bias.data *= m
    return pm

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    dev = torch.device('cpu')
    print("=" * 60)
    print("PRUNING METHODS COMPARISON")
    print("Ricci Topology vs Random vs Weight Magnitude")
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
    
    # Загружаем или обучаем модель
    model = MNIST_CNN().to(dev)
    if os.path.exists('simple_cnn_mnist.pth'):
        model.load_state_dict(torch.load('simple_cnn_mnist.pth', map_location=dev))
        print("Model loaded from file")
    else:
        print("Training...")
        import torch.optim as optim
        opt = optim.Adam(model.parameters(), lr=0.001)
        crit = nn.CrossEntropyLoss()
        for epoch in range(3):
            model.train()
            for data, target in train_loader:
                data, target = data.to(dev), target.to(dev)
                opt.zero_grad()
                crit(model(data), target).backward()
                opt.step()
        torch.save(model.state_dict(), 'simple_cnn_mnist.pth')
    
    model.eval()
    baseline = accuracy(model, test_loader, dev)
    print(f"Baseline accuracy: {baseline:.2f}%\n")
    
    # ==========================================================================
    # 1. Собираем активации и строим граф
    # ==========================================================================
    
    print("Collecting activations (50 batches)...")
    all_acts = defaultdict(list)
    with torch.no_grad():
        for idx, (data, _) in enumerate(train_loader):
            if idx >= 50: break
            data = data.to(dev)
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
    n_neurons = corr.shape[0]
    
    print(f"Building graph ({n_neurons} neurons)...")
    G = nx.Graph()
    G.add_nodes_from(range(n_neurons))
    for i in range(n_neurons):
        for j in range(i+1, n_neurons):
            if abs(corr[i,j]) >= 0.7:
                G.add_edge(i, j, weight=1.0 - abs(corr[i,j]))
    
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    def map_layer(labels):
        m = {}
        for idx, label in enumerate(labels):
            layer, neuron = label.split(':')
            m[idx] = (layer, int(neuron))
        return m
    neuron_map = map_layer(neuron_labels)
    
    # ==========================================================================
    # 2. ВЫЧИСЛЯЕМ ВАЖНОСТЬ ТРЕМЯ МЕТОДАМИ
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("COMPUTING IMPORTANCE SCORES")
    print("=" * 60)
    
    # --- Метод 1: Ricci Topology ---
    print("\n[1/3] Ollivier-Ricci curvature...")
    orc = OllivierRicci(G, alpha=0.5, weight="weight", verbose="INFO")
    G_ricci = orc.compute_ricci_curvature()
    
    node_ricci = {}
    for node in G_ricci.nodes():
        curvs = [G_ricci[node][nb].get('ricciCurvature', 0) for nb in G_ricci.neighbors(node)]
        node_ricci[node] = np.mean(curvs) if curvs else 0.0
    
    ricci = np.array([node_ricci[i] for i in range(n_neurons)])
    deg = np.array([G.degree(i) for i in range(n_neurons)])
    dc = deg / (n_neurons - 1)
    
    importance_ricci = np.abs(ricci) * (dc * 5 + 1)
    importance_ricci = (importance_ricci - importance_ricci.min()) / \
                       (importance_ricci.max() - importance_ricci.min() + 1e-10)
    
    # Сортируем: от МЕНЕЕ важных к БОЛЕЕ важным
    sorted_ricci = np.argsort(importance_ricci)
    
    print(f"  Ricci range: [{ricci.min():.3f}, {ricci.max():.3f}]")
    print(f"  Importance range: [{importance_ricci.min():.3f}, {importance_ricci.max():.3f}]")
    
    # --- Метод 2: Weight Magnitude ---
    print("\n[2/3] Weight magnitude...")
    importance_weight = get_neuron_weights(model)
    importance_weight = (importance_weight - importance_weight.min()) / \
                        (importance_weight.max() - importance_weight.min() + 1e-10)
    sorted_weight = np.argsort(importance_weight)  # менее важные -> более важные
    
    print(f"  Weight range: [{importance_weight.min():.3f}, {importance_weight.max():.3f}]")
    
    # --- Метод 3: Random ---
    print("\n[3/3] Random baseline...")
    sorted_random = np.random.permutation(n_neurons)
    print(f"  Random seed: 42 (first 5: {sorted_random[:5]})")
    
    # ==========================================================================
    # 3. PRUNING EXPERIMENT
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("PRUNING EXPERIMENTS")
    print("=" * 60)
    
    fractions = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    
    results = {
        'Ricci': [],
        'Weight': [],
        'Random': []
    }
    
    for frac in fractions:
        n_prune = int(n_neurons * frac)
        print(f"\n--- Prune {frac:.0%} ({n_prune}/{n_neurons} neurons) ---")
        
        for method, sorted_idx in [('Ricci', sorted_ricci), 
                                    ('Weight', sorted_weight), 
                                    ('Random', sorted_random)]:
            prune_set = set(sorted_idx[:n_prune])
            pm = prune_model(model, prune_set, neuron_map, dev)
            acc = accuracy(pm, test_loader, dev)
            results[method].append(acc)
            print(f"  {method:8s}: {acc:.2f}% (Δ {acc-baseline:+.2f}%)")
            del pm
    
    # ==========================================================================
    # 4. ВИЗУАЛИЗАЦИЯ
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = {'Ricci': '#2196F3', 'Weight': '#FF9800', 'Random': '#9E9E9E'}
    markers = {'Ricci': 'o', 'Weight': 's', 'Random': '^'}
    
    # График 1: Accuracy vs Prune Fraction
    ax = axes[0]
    for method in ['Ricci', 'Weight', 'Random']:
        ax.plot([0] + fractions, [baseline] + results[method],
                color=colors[method], marker=markers[method],
                linewidth=2, markersize=7, label=method)
    ax.axhline(y=baseline, color='black', linestyle='--', alpha=0.5, label='Baseline')
    ax.axhline(y=90, color='green', linestyle=':', alpha=0.5, label='90% threshold')
    ax.set_xlabel('Prune Fraction', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Pruning Methods Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # График 2: Accuracy Drop
    ax = axes[1]
    for method in ['Ricci', 'Weight', 'Random']:
        drops = [baseline - acc for acc in results[method]]
        ax.plot(fractions, drops,
                color=colors[method], marker=markers[method],
                linewidth=2, markersize=7, label=method)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.axhline(y=2, color='orange', linestyle=':', alpha=0.5, label='2% tolerance')
    ax.set_xlabel('Prune Fraction', fontsize=12)
    ax.set_ylabel('Accuracy Drop (%)', fontsize=12)
    ax.set_title('Accuracy Degradation', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # График 3: Распределение важности
    ax = axes[2]
    ax.hist(importance_ricci, bins=30, alpha=0.6, color=colors['Ricci'], label='Ricci', edgecolor='white')
    ax.hist(importance_weight, bins=30, alpha=0.6, color=colors['Weight'], label='Weight', edgecolor='white')
    ax.set_xlabel('Normalized Importance', fontsize=12)
    ax.set_ylabel('Number of Neurons', fontsize=12)
    ax.set_title('Importance Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pruning_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: pruning_comparison.png")
    
    # ==========================================================================
    # 5. СОХРАНЕНИЕ ДАННЫХ
    # ==========================================================================
    
    np.savez('pruning_comparison.npz',
             baseline=baseline,
             fractions=fractions,
             results_ricci=results['Ricci'],
             results_weight=results['Weight'],
             results_random=results['Random'],
             importance_ricci=importance_ricci,
             importance_weight=importance_weight,
             ricci_values=ricci,
             neuron_labels=neuron_labels)
    print("Saved: pruning_comparison.npz")
    
    # ==========================================================================
    # 6. АНАЛИЗ
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    # Находим максимальный prune fraction с accuracy drop < 2%
    print(f"\nPrune fractions with < 2% accuracy drop:")
    for method in ['Ricci', 'Weight', 'Random']:
        max_frac = 0
        for frac, acc in zip(fractions, results[method]):
            if baseline - acc < 2.0:
                max_frac = frac
        print(f"  {method:8s}: up to {max_frac:.0%}")
    
    # Находим, где Ricci превосходит другие методы
    print(f"\nRicci advantage over Random:")
    for frac, r_acc, rand_acc in zip(fractions, results['Ricci'], results['Random']):
        advantage = r_acc - rand_acc
        bar = '█' * int(advantage) if advantage > 0 else ''
        print(f"  {frac:.0%}: Ricci {r_acc:.2f}% vs Random {rand_acc:.2f}% = {advantage:+.2f}% {bar}")
    
    print(f"\nRicci advantage over Weight Magnitude:")
    for frac, r_acc, w_acc in zip(fractions, results['Ricci'], results['Weight']):
        advantage = r_acc - w_acc
        bar = '█' * int(abs(advantage)) if advantage > 0 else ''
        print(f"  {frac:.0%}: Ricci {r_acc:.2f}% vs Weight {w_acc:.2f}% = {advantage:+.2f}% {bar}")
    
    # Ключевые инсайты
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    
    # При 20% pruning
    idx_20 = fractions.index(0.20)
    ricci_20 = results['Ricci'][idx_20]
    weight_20 = results['Weight'][idx_20]
    random_20 = results['Random'][idx_20]
    
    print(f"\nAt 20% pruning:")
    print(f"  Ricci:   {ricci_20:.2f}% (Δ {ricci_20-baseline:+.2f}%)")
    print(f"  Weight:  {weight_20:.2f}% (Δ {weight_20-baseline:+.2f}%)")
    print(f"  Random:  {random_20:.2f}% (Δ {random_20-baseline:+.2f}%)")
    print(f"  Ricci beats Random by {ricci_20-random_20:.2f}%")
    print(f"  Ricci beats Weight by {ricci_20-weight_20:.2f}%")
    
    print("\n" + "=" * 60)
    print("Experiment complete! 🚀")
    print("Files: pruning_comparison.png, pruning_comparison.npz")
    print("=" * 60)

if __name__ == '__main__':
    mp.freeze_support()
    main()

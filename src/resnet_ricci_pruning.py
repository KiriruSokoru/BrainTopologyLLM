#!/usr/bin/env python3
"""
ResNet-18 Ricci Pruning (CIFAR-10) — v2 (совместимая загрузка)
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import os, warnings
warnings.filterwarnings('ignore')

torch.set_num_threads(4)

# ==============================================================================
# МОДЕЛЬ (такая же как при обучении)
# ==============================================================================

def create_model():
    model = models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

class ResNetWithHooks(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        self.model = create_model()
        if pretrained_path:
            self.model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
        
        self.activations = {}
        self._register_hooks()
    
    def _register_hooks(self):
        def hook(name):
            def fn(module, input, output):
                self.activations[name] = output.detach()
            return fn
        self.model.layer1.register_forward_hook(hook('layer1'))
        self.model.layer2.register_forward_hook(hook('layer2'))
        self.model.layer3.register_forward_hook(hook('layer3'))
        self.model.layer4.register_forward_hook(hook('layer4'))
    
    def forward(self, x):
        return self.model(x)

# ==============================================================================
# УТИЛИТЫ
# ==============================================================================

def accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return 100. * correct / total

def prune_model(model, prune_indices, neuron_map, device):
    """Обнуляет каналы в layer1-layer4."""
    layer_masks = defaultdict(list)
    for idx in prune_indices:
        layer, neuron = neuron_map[idx]
        layer_masks[layer].append(neuron)
    
    pm = ResNetWithHooks('resnet18_cifar10.pth').to(device)
    
    with torch.no_grad():
        for layer_name, neuron_list in layer_masks.items():
            layer = getattr(pm.model, layer_name)
            for neuron_idx in neuron_list:
                for child in layer.modules():
                    if isinstance(child, nn.Conv2d):
                        if child.weight.shape[0] > neuron_idx:
                            child.weight.data[neuron_idx] = 0
                            if child.bias is not None:
                                child.bias.data[neuron_idx] = 0
    return pm

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    device = torch.device('cpu')
    print("=" * 60)
    print("ResNet-18 Ricci Pruning (CIFAR-10)")
    print(f"Device: CPU, Threads: {torch.get_num_threads()}")
    print("=" * 60)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    test_set = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
    train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
    
    if not os.path.exists('resnet18_cifar10.pth'):
        print("resnet18_cifar10.pth not found!")
        return
    
    model = ResNetWithHooks('resnet18_cifar10.pth').to(device)
    model.eval()
    baseline = accuracy(model, test_loader, device)
    print(f"Baseline accuracy: {baseline:.2f}%")
    
    # Сбор активаций
    print("\nCollecting activations (30 batches)...")
    all_acts = defaultdict(list)
    with torch.no_grad():
        for idx, (data, _) in enumerate(train_loader):
            if idx >= 30: break
            data = data.to(device)
            _ = model(data)
            for name, act in model.activations.items():
                act_pooled = act.mean(dim=[2, 3])
                all_acts[name].append(act_pooled.cpu().numpy())
    
    for name in all_acts:
        all_acts[name] = np.concatenate(all_acts[name], axis=0)
    
    total_neurons = 0
    for name, acts in all_acts.items():
        print(f"  {name}: {acts.shape}")
        total_neurons += acts.shape[1]
    print(f"Total neurons: {total_neurons}")
    
    # Граф
    print("\nBuilding graph...")
    neuron_data = []
    neuron_labels = []
    for layer_name, acts in all_acts.items():
        for i in range(acts.shape[1]):
            neuron_data.append(acts[:, i])
            neuron_labels.append(f"{layer_name}:{i}")
    
    neuron_data = np.array(neuron_data)
    corr = np.corrcoef(neuron_data)
    n = corr.shape[0]
    
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1, n):
            if abs(corr[i, j]) >= 0.7:
                G.add_edge(i, j, weight=1.0 - abs(corr[i, j]))
    
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
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
    
    deg = np.array([G.degree(i) for i in range(n)])
    dc = np.where(n > 1, deg / (n - 1), 0)
    importance = np.abs(ricci) * (dc * 5 + 1)
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-10)
    
    sorted_ricci = np.argsort(importance)
    sorted_random = np.random.permutation(n)
    
    print("\nComputing weight importance...")
    weight_imp = []
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(model.model, layer_name)
        for child in layer.modules():
            if isinstance(child, nn.Conv2d):
                w = child.weight.data.abs().sum(dim=[1,2,3]).cpu().numpy()
                weight_imp.extend(w)
                break
    weight_imp = np.array(weight_imp[:n])
    weight_imp = (weight_imp - weight_imp.min()) / (weight_imp.max() - weight_imp.min() + 1e-10)
    sorted_weight = np.argsort(weight_imp)
    
    def map_layer(labels):
        m = {}
        for idx, label in enumerate(labels):
            layer, neuron = label.split(':')
            m[idx] = (layer, int(neuron))
        return m
    neuron_map = map_layer(neuron_labels)
    
    # Pruning
    print("\n" + "=" * 60)
    print("PRUNING")
    print("=" * 60)
    
    fractions = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results = {'Ricci': [], 'Weight': [], 'Random': []}
    
    for frac in fractions:
        n_prune = int(n * frac)
        print(f"\n--- Prune {frac:.0%} ({n_prune}/{n}) ---")
        
        for method, sorted_idx in [('Ricci', sorted_ricci),
                                    ('Weight', sorted_weight),
                                    ('Random', sorted_random)]:
            prune_set = set(sorted_idx[:n_prune])
            pm = prune_model(model, prune_set, neuron_map, device)
            acc = accuracy(pm, test_loader, device)
            results[method].append(acc)
            print(f"  {method:8s}: {acc:.2f}% (Δ {acc-baseline:+.2f}%)")
            del pm
    
    # Графики
    print("\nSaving...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {'Ricci': '#2196F3', 'Weight': '#FF9800', 'Random': '#9E9E9E'}
    
    ax = axes[0]
    for method in ['Ricci', 'Weight', 'Random']:
        ax.plot([0] + fractions, [baseline] + results[method],
                color=colors[method], marker='o', linewidth=2, markersize=7, label=method)
    ax.axhline(y=baseline, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Prune Fraction')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('ResNet-18 on CIFAR-10: Pruning Methods')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.hist(ricci, bins=40, color='steelblue', edgecolor='white', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Ricci Curvature')
    ax.set_ylabel('Neurons')
    neg = np.sum(ricci < -1e-6)
    zero = np.sum(np.abs(ricci) < 1e-6)
    pos = np.sum(ricci > 1e-6)
    ax.set_title(f'Ricci Distribution (Neg:{neg} Zero:{zero} Pos:{pos})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/resnet_pruning.png', dpi=150)
    print("Saved: results/resnet_pruning.png")
    
    np.savez('results/resnet_pruning.npz',
             baseline=baseline, fractions=fractions,
             r=results['Ricci'], w=results['Weight'], rnd=results['Random'],
             ricci=ricci)
    
    print("\n" + "=" * 60)
    print(f"Baseline: {baseline:.2f}% | Graph: {n} nodes, {G.number_of_edges()} edges")
    print(f"Ricci: [{ricci.min():.4f}, {ricci.max():.4f}]")
    for frac, r, w, rnd in zip(fractions, results['Ricci'], results['Weight'], results['Random']):
        print(f"  {frac:.0%}: Ricci={r:.2f}% Weight={w:.2f}% Random={rnd:.2f}%")
    print("Done!")

if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    main()

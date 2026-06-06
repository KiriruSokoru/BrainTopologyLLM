#!/usr/bin/env python3
"""
MNIST Ricci Flow — Linux native (fork works here)
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
import os, warnings
warnings.filterwarnings('ignore')

# Ограничиваем потоки
torch.set_num_threads(4)
os.environ['OMP_NUM_THREADS'] = '4'

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

def accuracy(m, loader):
    m.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = m(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100. * correct / len(loader.dataset)

def main():
    global device
    device = torch.device('cpu')
    print("=" * 50)
    print("MNIST Ricci Flow — Linux Native")
    print(f"Device: CPU, Threads: {torch.get_num_threads()}")
    print("=" * 50)
    
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
    if os.path.exists('simple_cnn_mnist.pth'):
        model.load_state_dict(torch.load('simple_cnn_mnist.pth', map_location=device))
        print("Model loaded from file")
    else:
        print("Training model (2 min)...")
        import torch.optim as optim
        opt = optim.Adam(model.parameters(), lr=0.001)
        crit = nn.CrossEntropyLoss()
        for epoch in range(3):
            model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                opt.zero_grad()
                loss = crit(model(data), target)
                loss.backward()
                opt.step()
            print(f"  Epoch {epoch+1}/3 done")
        torch.save(model.state_dict(), 'simple_cnn_mnist.pth')
        print("  Saved: simple_cnn_mnist.pth")
    
    model.eval()
    baseline = accuracy(model, test_loader)
    print(f"Baseline accuracy: {baseline:.2f}%")
    
    # Активации (50 батчей)
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
    
    # Граф
    print("Building graph...")
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
            if abs(corr[i,j]) >= 0.7:
                G.add_edge(i, j, weight=1.0 - abs(corr[i,j]))
    
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Ricci (использует fork — на Linux работает)
    print("\nComputing Ollivier-Ricci curvature...")
    print("(This takes ~1-2 min, using all cores)")
    
    orc = OllivierRicci(G, alpha=0.5, weight="weight", verbose="INFO")
    G_ricci = orc.compute_ricci_curvature()
    
    node_ricci = {}
    for node in G_ricci.nodes():
        curvs = [G_ricci[node][nb].get('ricciCurvature', 0) for nb in G_ricci.neighbors(node)]
        node_ricci[node] = np.mean(curvs) if curvs else 0.0
    
    ricci = np.array([node_ricci[i] for i in range(n)])
    print(f"Ricci: min={ricci.min():.4f}, max={ricci.max():.4f}, mean={ricci.mean():.4f}")
    
    # Важность
    deg = np.array([G.degree(i) for i in range(n)])
    dc = deg / (n-1)
    importance = np.abs(ricci) * (dc * 5 + 1)
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-10)
    
    # Pruning
    print("\nPruning experiments...")
    sorted_idx = np.argsort(importance)
    
    def map_layer(labels):
        m = {}
        for idx, label in enumerate(labels):
            layer, neuron = label.split(':')
            m[idx] = (layer, int(neuron))
        return m
    
    neuron_map = map_layer(neuron_labels)
    fractions = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = []
    
    for frac in fractions:
        n_prune = int(n * frac)
        prune_set = set(sorted_idx[:n_prune])
        
        masks = defaultdict(lambda: torch.ones(200))
        for idx in prune_set:
            layer, neuron = neuron_map[idx]
            masks[layer][neuron] = 0.0
        
        pm = MNIST_CNN().to(device)
        pm.load_state_dict(model.state_dict())
        
        with torch.no_grad():
            if 'conv1' in masks:
                m = masks['conv1'][:16].to(device)
                pm.conv1.weight.data *= m.view(1,-1,1,1)
                pm.conv1.bias.data *= m
            if 'conv2' in masks:
                m = masks['conv2'][:32].to(device)
                pm.conv2.weight.data *= m.view(1,-1,1,1)
                pm.conv2.bias.data *= m
            if 'fc1' in masks:
                m = masks['fc1'][:128].to(device)
                pm.fc1.weight.data *= m.unsqueeze(1)
                pm.fc1.bias.data *= m
        
        acc = accuracy(pm, test_loader)
        results.append(acc)
        print(f"  Prune {frac:.0%}: {acc:.2f}% (drop {baseline-acc:.2f}%)")
        del pm
    
    # Сохранение
    print("\nSaving results...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot([0] + fractions, [baseline] + results, 'bo-', lw=2, ms=8)
    axes[0].axhline(y=baseline, color='r', linestyle='--', label='Baseline')
    axes[0].set_xlabel('Prune Fraction')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Topology-based Pruning (Ollivier-Ricci)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(ricci, bins=30, color='steelblue', edgecolor='white')
    axes[1].axvline(x=0, color='red', linestyle='--')
    axes[1].set_xlabel('Ricci Curvature')
    axes[1].set_title('Ricci Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ricci_results_linux.png', dpi=150)
    print("Saved: ricci_results_linux.png")
    
    np.savez('ricci_results_linux.npz', baseline=baseline, fractions=fractions,
             results=results, ricci=ricci, importance=importance)
    
    # Итоги
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Baseline: {baseline:.2f}%")
    print(f"Ricci: [{ricci.min():.4f}, {ricci.max():.4f}]")
    neg = np.sum(ricci < -1e-6)
    zero = np.sum(np.abs(ricci) < 1e-6)
    pos = np.sum(ricci > 1e-6)
    print(f"Neg/Zero/Pos: {neg}/{zero}/{pos}")
    
    for frac, acc in zip(fractions, results):
        loss = baseline - acc
        s = "OK" if loss < 2 else "WARN" if loss < 5 else "FAIL"
        print(f"  [{s}] {frac:.0%}: {acc:.2f}% (loss {loss:.2f}%)")
    
    print("\nDone! 🚀")

if __name__ == '__main__':
    mp.freeze_support()
    main()

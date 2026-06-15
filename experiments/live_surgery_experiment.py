"""
Эксперимент: живая хирургия Перельмана в реальном времени.
Версия 2.2: ОТЛАДКА — добавлен вывод статистики графа и кривизны.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm
import random

# ========== НАСТРОЙКИ ==========
SEEDS = [42, 123, 456, 789, 1011]
EPOCHS = 30
SURGERY_INTERVAL = 5
HIDDEN_SIZE = 64
BATCH_SIZE = 256
LEARNING_RATE = 0.01
RICCI_THRESHOLD = 0.0  # ИСПРАВЛЕНО: было -0.1, теперь 0.0 (ловим всё отрицательное)
GRAPH_THRESHOLD = 0.1  # ИСПРАВЛЕНО: было 0.3, теперь 0.1 (больше рёбер)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = 'results/checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"Используем устройство: {DEVICE}")


# ========== СЕТЬ ==========
class SimpleMLP(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x, return_activations=False):
        x = x.view(-1, 784)
        a1 = self.relu(self.fc1(x))
        a2 = self.relu(self.fc2(a1))
        out = self.fc3(a2)
        if return_activations:
            return out, {'fc1': a1, 'fc2': a2}
        return out


# ========== ДАННЫЕ ==========
def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


# ========== ОБУЧЕНИЕ И ОЦЕНКА ==========
def train_one_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data, target in train_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    return total_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return total_loss / len(test_loader), 100. * correct / total


# ========== ГРАФ И КРИВИЗНА ==========
def collect_activations(model, train_loader, n_batches=10):
    model.eval()
    activations = {'fc1': [], 'fc2': []}
    with torch.no_grad():
        for i, (data, _) in enumerate(train_loader):
            if i >= n_batches: break
            data = data.to(DEVICE)
            _, acts = model(data, return_activations=True)
            for key in activations:
                activations[key].append(acts[key].cpu().numpy())
    for key in activations:
        activations[key] = np.concatenate(activations[key], axis=0)
    return activations

def build_correlation_graph(activations, threshold=GRAPH_THRESHOLD):
    n_neurons = activations.shape[1]
    G = nx.Graph()
    G.add_nodes_from(range(n_neurons))
    
    # ИСПРАВЛЕНО: Обрабатываем "мёртвые" нейроны заранее
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr_matrix = np.corrcoef(activations.T)
    
    # Заменяем NaN и inf на 0
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            if abs(corr_matrix[i, j]) > threshold:
                G.add_edge(i, j, weight=abs(corr_matrix[i, j]))
    
    # ОТЛАДКА: Выводим статистику графа
    print(f"    [ГРАФ] Узлов: {len(G.nodes)}, Рёбер: {len(G.edges)}, Плотность: {nx.density(G):.3f}")
    
    return G

def compute_ricci_curvature(G):
    if len(G.edges) == 0:
        print("    [КРИВИЗНА] Граф пустой, все кривизны = 0")
        return {node: 0.0 for node in G.nodes}
    
    orc = OllivierRicci(G, alpha=0.5)
    G_with_ricci = orc.compute_ricci_curvature()
    
    curvatures = {node: G_with_ricci.nodes[node].get('ricciCurvature', 0.0) for node in G.nodes}
    
    # ОТЛАДКА: Выводим статистику кривизны
    curv_values = list(curvatures.values())
    print(f"    [КРИВИЗНА] Min: {min(curv_values):.4f}, Max: {max(curv_values):.4f}, Mean: {np.mean(curv_values):.4f}")
    print(f"    [КРИВИЗНА] Отрицательных: {sum(1 for v in curv_values if v < 0)}, Положительных: {sum(1 for v in curv_values if v > 0)}")
    
    return curvatures

def find_singularities(curvatures, threshold=RICCI_THRESHOLD):
    singularities = [node for node, curv in curvatures.items() if curv < threshold]
    print(f"    [СИНГУЛЯРНОСТИ] Найдено: {len(singularities)} (порог: {threshold})")
    return singularities

def find_masters(curvatures, threshold=0.1):
    masters = [node for node, curv in curvatures.items() if curv > threshold]
    print(f"    [МАСТЕРА] Найдено: {len(masters)} (порог: {threshold})")
    return masters


# ========== ХИРУРГИЯ ==========
def perform_ricci_surgery(model, layer_name='fc2', train_loader=None):
    print(f"  [ХИРУРГИЯ] Начинаем диагностику слоя {layer_name}")
    
    activations = collect_activations(model, train_loader)
    layer_acts = activations[layer_name]
    
    print(f"  [АКТИВАЦИИ] Shape: {layer_acts.shape}, Mean: {layer_acts.mean():.4f}, Std: {layer_acts.std():.4f}")
    
    G = build_correlation_graph(layer_acts)
    curvatures = compute_ricci_curvature(G)
    
    singularities = find_singularities(curvatures)
    masters = find_masters(curvatures)
    
    if len(singularities) == 0 or len(masters) == 0:
        print(f"  [ХИРУРГИЯ] Пропуск: сингулярностей={len(singularities)}, мастеров={len(masters)}")
        return len(singularities), len(masters)
    
    layer = getattr(model, layer_name)
    weights = layer.weight.data.clone()
    master_weights = weights[masters].mean(dim=0)
    
    for sing in singularities:
        weights[sing] = master_weights
    layer.weight.data = weights
    
    print(f"  [ХИРУРГИЯ] Заменено {len(singularities)} сингулярностей на среднее {len(masters)} мастеров")
    return len(singularities), len(masters)

def perform_random_surgery(model, layer_name='fc2', n_replace=5):
    layer = getattr(model, layer_name)
    weights = layer.weight.data.clone()
    n_neurons = weights.shape[0]
    targets = random.sample(range(n_neurons), min(n_replace, n_neurons))
    sources = random.sample(range(n_neurons), min(n_replace, n_neurons))
    for t, s in zip(targets, sources):
        weights[t] = weights[s]
    layer.weight.data = weights
    return len(targets)

def perform_magnitude_surgery(model, layer_name='fc2', n_replace=5):
    layer = getattr(model, layer_name)
    weights = layer.weight.data.clone()
    importance = weights.abs().sum(dim=1)
    _, least_important = importance.topk(n_replace, largest=False)
    _, most_important = importance.topk(n_replace, largest=True)
    for target, source in zip(least_important, most_important):
        weights[target] = weights[source]
    layer.weight.data = weights
    return n_replace


# ========== ЧЕКПОИНТЫ ==========
def get_checkpoint_path(strategy, seed):
    return os.path.join(CHECKPOINT_DIR, f"{strategy}_seed{seed}.pt")

def save_checkpoint(model, optimizer, epoch, history, strategy, seed):
    path = get_checkpoint_path(strategy, seed)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'strategy': strategy,
        'seed': seed
    }, path)

def load_checkpoint(model, optimizer, strategy, seed):
    path = get_checkpoint_path(strategy, seed)
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['history']
    return 0, {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'singularities': [], 'epoch': []
    }


# ========== ОДИН ЗАПУСК С ЧЕКПОИНТАМИ ==========
def run_single_experiment(seed, strategy, train_loader, test_loader):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    model = SimpleMLP(HIDDEN_SIZE).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    start_epoch, history = load_checkpoint(model, optimizer, strategy, seed)
    
    if start_epoch > 0:
        print(f"  [ВОЗОБНОВЛЕНИЕ] Найдён чекпоинт для seed={seed}. Продолжаем с эпохи {start_epoch + 1}")
    else:
        print(f"  [НОВЫЙ ЗАПУСК] seed={seed}")

    pbar = tqdm(range(start_epoch + 1, EPOCHS + 1), desc=f"[{strategy}] seed={seed}")
    
    for epoch in pbar:
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch'].append(epoch)
        
        n_sing = 0
        if strategy == 'live_ricci' and epoch % SURGERY_INTERVAL == 0 and epoch < EPOCHS:
            n_sing, _ = perform_ricci_surgery(model, 'fc2', train_loader)
        elif strategy == 'live_random' and epoch % SURGERY_INTERVAL == 0 and epoch < EPOCHS:
            n_sing = perform_random_surgery(model, 'fc2', n_replace=5)
        elif strategy == 'live_magnitude' and epoch % SURGERY_INTERVAL == 0 and epoch < EPOCHS:
            n_sing = perform_magnitude_surgery(model, 'fc2', n_replace=5)
        elif strategy == 'posthoc' and epoch == EPOCHS - 1:
            n_sing, _ = perform_ricci_surgery(model, 'fc2', train_loader)
        
        history['singularities'].append(n_sing)
        
        pbar.set_postfix({
            'loss': f"{test_loss:.4f}",
            'acc': f"{test_acc:.1f}%",
            'sing': n_sing
        })
        
        save_checkpoint(model, optimizer, epoch, history, strategy, seed)
    
    return history


# ========== ГЛАВНАЯ ФУНКЦИЯ ==========
def run_experiment():
    train_loader, test_loader = get_dataloaders()
    strategies = ['live_ricci', 'live_random', 'live_magnitude']  # Убрал baseline и posthoc — они уже есть
    all_results = {}
    
    # Загружаем результаты baseline и posthoc из чекпоинтов
    for strategy in ['baseline', 'posthoc']:
        strategy_results = []
        for seed in SEEDS:
            path = get_checkpoint_path(strategy, seed)
            if os.path.exists(path):
                checkpoint = torch.load(path, map_location=DEVICE)
                strategy_results.append(checkpoint['history'])
        all_results[strategy] = strategy_results
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Стратегия: {strategy.upper()}")
        print(f"{'='*60}")
        
        strategy_results = []
        for seed in SEEDS:
            history = run_single_experiment(seed, strategy, train_loader, test_loader)
            strategy_results.append(history)
        
        all_results[strategy] = strategy_results
    
    os.makedirs('results', exist_ok=True)
    with open('results/live_surgery_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n✅ Результаты и чекпоинты сохранены.")
    plot_results(all_results)
    
    return all_results


# ========== ГРАФИКИ ==========
def plot_results(all_results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    strategies_names = {
        'baseline': 'Baseline',
        'posthoc': 'Post-hoc',
        'live_ricci': 'Live Ricci',
        'live_random': 'Random',
        'live_magnitude': 'Magnitude'
    }
    
    colors = {
        'baseline': 'gray',
        'posthoc': 'blue',
        'live_ricci': 'red',
        'live_random': 'green',
        'live_magnitude': 'orange'
    }
    
    ax = axes[0, 0]
    for strategy, results in all_results.items():
        mean_loss = np.mean([r['test_loss'] for r in results], axis=0)
        std_loss = np.std([r['test_loss'] for r in results], axis=0)
        epochs = results[0]['epoch']
        ax.plot(epochs, mean_loss, label=strategies_names[strategy], color=colors[strategy], linewidth=2)
        ax.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, color=colors[strategy], alpha=0.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Loss')
    ax.set_title('Loss Convergence (mean ± std)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for strategy, results in all_results.items():
        mean_acc = np.mean([r['test_acc'] for r in results], axis=0)
        std_acc = np.std([r['test_acc'] for r in results], axis=0)
        epochs = results[0]['epoch']
        ax.plot(epochs, mean_acc, label=strategies_names[strategy], color=colors[strategy], linewidth=2)
        ax.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc, color=colors[strategy], alpha=0.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Accuracy Growth (mean ± std)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    for strategy, results in all_results.items():
        var_by_epoch = []
        for i in range(len(results[0]['epoch'])):
            losses_at_epoch = [r['test_loss'][i] for r in results]
            var_by_epoch.append(np.var(losses_at_epoch))
        epochs = results[0]['epoch']
        ax.plot(epochs, var_by_epoch, label=strategies_names[strategy], color=colors[strategy], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Variance')
    ax.set_title('Collapse to Attractor (Variance between seeds)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    ax = axes[1, 1]
    strategies_list = list(all_results.keys())
    final_accs = [np.mean([r['test_acc'][-1] for r in all_results[s]]) for s in strategies_list]
    final_vars = [np.var([r['test_loss'][-1] for r in all_results[s]]) for s in strategies_list]
    
    x = np.arange(len(strategies_list))
    width = 0.35
    ax.bar(x - width/2, final_accs, width, label='Final Acc (%)', color='steelblue')
    ax2 = ax.twinx()
    ax2.bar(x + width/2, final_vars, width, label='Final Var', color='coral')
    
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Accuracy (%)', color='steelblue')
    ax2.set_ylabel('Variance', color='coral')
    ax.set_title('Final Results')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies_list, fontsize=9)
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('results/live_surgery_comparison.png', dpi=150, bbox_inches='tight')
    print("\n📊 Графики сохранены в results/live_surgery_comparison.png")
    plt.show()


# ========== ТОЧКА ВХОДА ==========
if __name__ == '__main__':
    import warnings
    print("="*60)
    print("ЭКСПЕРИМЕНТ: ЖИВАЯ ХИРУРГИЯ ПЕРЕЛЬМАНА (v2.2 ОТЛАДКА)")
    print("="*60)
    run_experiment()

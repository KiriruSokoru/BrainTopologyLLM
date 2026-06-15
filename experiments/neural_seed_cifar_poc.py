import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import warnings
import os

# ========== НАСТРОЙКИ ==========
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS_TRAIN = 10
NUM_RUNS = 3

print(f"Устройство: {DEVICE}")
os.makedirs("results", exist_ok=True)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ========== ДАННЫЕ ==========
def get_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10('./data', train=False, transform=transform)
    
    train_subset = torch.utils.data.Subset(train_ds, range(3000))
    test_subset = torch.utils.data.Subset(test_ds, range(1000))
    
    return DataLoader(train_subset, batch_size=64, shuffle=True), \
           DataLoader(test_subset, batch_size=64, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x.to(DEVICE)).argmax(1) == y.to(DEVICE)).sum().item()
    model.train()
    return 100. * correct / len(loader.dataset)

def train_model(model, train_loader, epochs):
    opt = optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in train_loader:
            opt.zero_grad()
            crit(model(x.to(DEVICE)), y.to(DEVICE)).backward()
            opt.step()

# ========== ТОПЛОГИЧЕСКИЙ АНАЛИЗ ==========
def get_masters(model, train_loader, top_k=30):
    model.eval()
    acts = []
    
    with torch.no_grad():
        for x, _ in train_loader:
            x = model.pool1(model.relu1(model.conv1(x.to(DEVICE))))
            x = model.pool2(model.relu2(model.conv2(x)))
            x = x.view(-1, 64 * 8 * 8)
            act = model.relu3(model.fc1(x))
            acts.append(act.cpu().numpy())
            if len(acts) >= 10:
                break
    
    acts = np.concatenate(acts, axis=0)
    print(f"    Размер активаций: {acts.shape}")
    
    G = nx.Graph()
    G.add_nodes_from(range(acts.shape[1]))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = np.nan_to_num(np.corrcoef(acts.T), nan=0.0)
    
    for i in range(acts.shape[1]):
        for j in range(i+1, acts.shape[1]):
            if abs(corr[i,j]) > 0.3:
                G.add_edge(i, j, weight=abs(corr[i,j]))
    
    if len(G.edges) == 0:
        print("    ⚠️ Граф пуст, возвращаем случайные нейроны")
        return list(range(min(top_k, acts.shape[1])))
    
    orc = OllivierRicci(G, alpha=0.5)
    Gr = orc.compute_ricci_curvature()
    curv = {n: Gr.nodes[n].get('ricciCurvature', 0.0) for n in G.nodes}
    
    sorted_masters = sorted(curv.items(), key=lambda x: x[1], reverse=True)
    top_masters = [n for n, c in sorted_masters[:top_k] if c > 0.05]
    
    if len(top_masters) < top_k:
        remaining = [n for n in range(acts.shape[1]) if n not in top_masters]
        top_masters.extend(remaining[:top_k - len(top_masters)])
    
    print(f"    Найдено мастеров с кривизной > 0.05: {len([n for n, c in curv.items() if c > 0.05])}")
    print(f"    Оставляем топ-{top_k}: {len(top_masters)}")
    
    return top_masters

def create_neural_seed(model, train_loader, top_k=30):
    masters = get_masters(model, train_loader, top_k=top_k)
    
    seed_data = {
        "architecture": "SimpleCNN",
        "master_indices": masters,
        "master_weights_fc1": model.fc1.weight.data[masters].clone().cpu(),
        "master_weights_fc2": model.fc2.weight.data[:, masters].clone().cpu(),
        "total_params_original": sum(p.numel() for p in model.parameters())
    }
    
    return seed_data

def grow_model_from_seed_with_checkpoints(seed_data, train_loader, test_loader, total_epochs):
    """Выращивает модель и измеряет точность после КАЖДОЙ эпохи."""
    new_model = SimpleCNN().to(DEVICE)
    
    if seed_data is not None:
        masters = seed_data["master_indices"]
        new_model.fc1.weight.data[masters] = seed_data["master_weights_fc1"].to(DEVICE)
        new_model.fc2.weight.data[:, masters] = seed_data["master_weights_fc2"].to(DEVICE)
    
    opt = optim.Adam(new_model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    
    accuracies_per_epoch = []
    
    for epoch in range(total_epochs):
        # Обучаем одну эпоху
        for x, y in train_loader:
            opt.zero_grad()
            crit(new_model(x.to(DEVICE)), y.to(DEVICE)).backward()
            opt.step()
        
        # Измеряем точность после этой эпохи
        acc = evaluate(new_model, test_loader)
        accuracies_per_epoch.append(acc)
    
    return new_model, accuracies_per_epoch

# ========== ГЛАВНЫЙ ПРОЦЕСС ==========
if __name__ == "__main__":
    print("="*70)
    print("NEURAL SEED: CIFAR-10 — Измерение СКОРОСТИ СХОДИМОСТИ")
    print("="*70)
    
    train_loader, test_loader = get_cifar10()
    
    print("\n[ЭТАП 1] Обучение эталонной модели...")
    set_seed(42)
    original_model = SimpleCNN().to(DEVICE)
    train_model(original_model, train_loader, EPOCHS_TRAIN)
    acc_original = evaluate(original_model, test_loader)
    print(f"✅ Эталонная точность: {acc_original:.2f}%")
    
    # Создаем семена
    print("\n[ЭТАП 2] Создание мастеров для разных top_k...")
    seeds = {}
    for k in [20, 10, 5]:
        seeds[k] = create_neural_seed(original_model, train_loader, top_k=k)
    
    # Множественные запуски с измерением точности после КАЖДОЙ эпохи
    top_k_values = [0, 20, 10, 5]
    results_per_epoch = {k: [] for k in top_k_values}
    
    for run in range(NUM_RUNS):
        print(f"\n{'='*70}")
        print(f"ЗАПУСК {run + 1}/{NUM_RUNS}")
        print(f"{'='*70}")
        
        set_seed(run * 100)
        
        for k in top_k_values:
            seed_data = seeds[k] if k > 0 else None
            _, accs = grow_model_from_seed_with_checkpoints(
                seed_data, train_loader, test_loader, total_epochs=5
            )
            results_per_epoch[k].append(accs)
            
            print(f"  top_k={k:<2}: после 1 эпохи = {accs[0]:.2f}%, после 5 эпох = {accs[-1]:.2f}%")
    
    # ИТОГИ: Скорость сходимости (точность после 1 эпохи)
    print("\n" + "="*70)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ: СКОРОСТЬ СХОДИМОСТИ (после 1 эпохи)")
    print("="*70)
    print(f"{'top_k':<8} | {'Mean':<8} | {'Std':<8} | {'Прирост':<10}")
    print("-" * 70)
    
    baseline_mean_epoch1 = np.mean([results_per_epoch[k][0][0] for k in [0] for run in results_per_epoch[k]])
    
    for k in top_k_values:
        accs_epoch1 = [results_per_epoch[k][run][0] for run in range(NUM_RUNS)]
        mean_acc = np.mean(accs_epoch1)
        std_acc = np.std(accs_epoch1)
        gain = mean_acc - baseline_mean_epoch1
        
        label = "baseline" if k == 0 else f"{k}"
        print(f"{label:<8} | {mean_acc:<8.2f} | {std_acc:<8.2f} | {gain:+.2f} п.п.")
    
    print("="*70)
    
    print("\nАНАЛИЗ:")
    print(f"1. Baseline после 1 эпохи: {baseline_mean_epoch1:.2f}%")
    
    print("\n2. Прирост от мастеров после 1 эпохи (скорость сходимости):")
    for k in [20, 10, 5]:
        accs_epoch1 = [results_per_epoch[k][run][0] for run in range(NUM_RUNS)]
        gain = np.mean(accs_epoch1) - baseline_mean_epoch1
        print(f"   top_k={k}: {gain:+.2f} п.п.")
    
    print("\n3. Сравнение с финальной точностью (после 5 эпох):")
    for k in top_k_values:
        accs_epoch5 = [results_per_epoch[k][run][-1] for run in range(NUM_RUNS)]
        mean_acc = np.mean(accs_epoch5)
        label = "baseline" if k == 0 else f"{k}"
        print(f"   {label:<8}: {mean_acc:.2f}%")
    
    print("\nВЫВОД:")
    print("Если мастера дают прирост после 1 эпохи, но не после 5,")
    print("значит, они работают как 'стартер' — ускоряют сходимость,")
    print("но не влияют на финальный результат.")
    print("="*70)

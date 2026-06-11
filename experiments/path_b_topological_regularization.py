"""
Сравнение стратегий регуляризации: Базовая, Мягкая (20%) и Жесткая (50%).
Поиск оптимального коэффициента топологической проекции (Эмбриогенез весов).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import warnings

# ========== НАСТРОЙКИ ==========
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_SIZE = 128
EPOCHS = 15
STEPS_PER_CHECK = 50  
NOISE_RATIO = 0.2  # 20% шума в лейблах

print(f"Устройство: {DEVICE}")

# ========== ДАННЫЕ С ШУМОМ ==========
def get_noisy_mnist():
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, transform=transform)
    
    X_train, y_train = train_ds.data[:5000].float().view(-1, 784) / 255.0, train_ds.targets[:5000]
    X_test, y_test = test_ds.data[:1000].float().view(-1, 784) / 255.0, test_ds.targets[:1000]
    
    num_noisy = int(len(y_train) * NOISE_RATIO)
    noisy_indices = torch.randperm(len(y_train))[:num_noisy]
    y_train_noisy = y_train.clone()
    y_train_noisy[noisy_indices] = torch.randint(0, 10, (num_noisy,))
    
    train_loader = DataLoader(TensorDataset(X_train, y_train_noisy), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)
    return train_loader, test_loader

# ========== МОДЕЛЬ ==========
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_SIZE, 10)
        
    def forward(self, x, return_act=False):
        act = self.relu(self.fc1(x))
        out = self.fc2(act)
        if return_act:
            return out, act
        return out

# ========== ТОПЛОГИЧЕСКАЯ РЕГУЛЯРИЗАЦИЯ ==========
def apply_topological_projection(model, train_loader, alpha):
    if alpha == 0.0:
        return 0
        
    model.eval()
    activations = []
    with torch.no_grad():
        for i, (x, _) in enumerate(train_loader):
            if i >= 5: break
            _, act = model(x.to(DEVICE), return_act=True)
            activations.append(act.cpu().numpy())
    
    acts = np.concatenate(activations, axis=0)
    n_neurons = acts.shape[1]
    G = nx.Graph()
    G.add_nodes_from(range(n_neurons))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = np.corrcoef(acts.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    
    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            if abs(corr[i, j]) > 0.3:
                G.add_edge(i, j, weight=abs(corr[i, j]))
                
    if len(G.edges) == 0:
        model.train()
        return 0
        
    orc = OllivierRicci(G, alpha=0.5)
    G_ricci = orc.compute_ricci_curvature()
    
    curvatures = {node: G_ricci.nodes[node].get('ricciCurvature', 0.0) for node in G.nodes}
    singularities = [node for node, c in curvatures.items() if c < 0.0]
    masters = [node for node, c in curvatures.items() if c > 0.1]
    
    suppressed_count = 0
    if len(singularities) > 0 and len(masters) > 0:
        weights = model.fc1.weight.data.clone()
        master_weights = weights[masters].mean(dim=0)
        
        for sing in singularities:
            # ФОРМУЛА ПРОЕКЦИИ: (1 - alpha) * текущий_вес + alpha * вес_мастера
            weights[sing] = (1 - alpha) * weights[sing] + alpha * master_weights
            suppressed_count += 1
            
        model.fc1.weight.data = weights
        
    model.train()
    return suppressed_count

# ========== ОБУЧЕНИЕ ==========
def train_model(name, alpha, train_loader, test_loader):
    print(f"\n{'='*60}")
    print(f"Запуск: {name} (Alpha = {alpha})")
    print(f"{'='*60}")
    
    model = SimpleMLP().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    step_count = 0
    last_sing_count = 0
    
    for epoch in range(EPOCHS):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            step_count += 1
            
            if alpha > 0.0 and step_count % STEPS_PER_CHECK == 0:
                last_sing_count = apply_topological_projection(model, train_loader, alpha)
                
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                out = model(x.to(DEVICE))
                correct += (out.argmax(1) == y.to(DEVICE)).sum().item()
                total += y.size(0)
        test_acc = 100. * correct / total
        model.train()
        
        if epoch % 3 == 0 or epoch == EPOCHS - 1:
            reg_status = f"| Подавлено сингулярностей: {last_sing_count}" if alpha > 0.0 else ""
            print(f"Эпоха {epoch:2d} | Test Acc: {test_acc:5.2f}% {reg_status}")
            
    return test_acc

# ========== ГЛАВНЫЙ ПРОЦЕСС ==========
if __name__ == "__main__":
    print("="*70)
    print("СРАВНЕНИЕ СТРАТЕГИЙ РЕГУЛЯРИЗАЦИИ (Эмбриогенез весов)")
    print(f"Условия: MNIST с {NOISE_RATIO*100:.0f}% шума в лейблах")
    print("="*70)
    
    train_loader, test_loader = get_noisy_mnist()
    
    # Путь 1: Без регуляризации
    acc_baseline = train_model("1. BASELINE (Без регуляризации)", alpha=0.0, train_loader=train_loader, test_loader=test_loader)
    
    # Путь 2: Мягкая регуляризация (20%)
    acc_soft = train_model("2. SOFT REG (Мягкая проекция, Alpha=0.2)", alpha=0.2, train_loader=train_loader, test_loader=test_loader)
    
    # Путь 3: Жесткая регуляризация (50%)
    acc_hard = train_model("3. HARD REG (Жесткая проекция, Alpha=0.5)", alpha=0.5, train_loader=train_loader, test_loader=test_loader)
    
    print("\n" + "="*70)
    print("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*70)
    print(f"1. Baseline (Alpha=0.0):  {acc_baseline:.2f}%")
    print(f"2. Soft Reg (Alpha=0.2):  {acc_soft:.2f}%  {'<-- Текущий лидер' if acc_soft > max(acc_baseline, acc_hard) else ''}")
    print(f"3. Hard Reg (Alpha=0.5):  {acc_hard:.2f}%  {'<-- Новый лидер!' if acc_hard > max(acc_baseline, acc_soft) else ''}")
    print("="*70)
    
    if acc_hard > acc_baseline and acc_hard > acc_soft:
        print("🚀 ПРОРЫВ: Жесткая топологическая регуляризация (50%) дала максимальную обобщающую способность!")
    elif acc_soft > acc_baseline and acc_soft > acc_hard:
        print("⚖️ БАЛАНС: Мягкая регуляризация (20%) является оптимальной. Жесткая (50%) привела к недообучению.")
    else:
        print("🔍 АНАЛИЗ: Любое топологическое вмешательство ухудшило результат. Требуется пересмотр порога графа или alpha.")
    print("="*70)

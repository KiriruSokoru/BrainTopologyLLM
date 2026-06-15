"""
Эксперимент: Поиск "Божьей Искры" v2 (Итеративная Проекция)
Проверяем, способны ли K замороженных мастеров итеративно вытащить разрушенную сеть к аттрактору.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import warnings
from tqdm import tqdm

# ========== НАСТРОЙКИ ==========
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_SIZE = 128
EPOCHS = 8
DAMAGE_SCALE = 5.0  # Сила шума при разрушении
ITERATIONS = 10     # Количество итераций хирургии
K_SWEEP = [1, 2, 3, 5, 10, 20]

print(f"Устройство: {DEVICE}")

# ========== ДАННЫЕ ==========
def get_mnist():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, transform=transform)
    
    X_train, y_train = train_ds.data[:2000].float().view(-1, 784) / 255.0, train_ds.targets[:2000]
    X_test, y_test = test_ds.data[:1000].float().view(-1, 784) / 255.0, test_ds.targets[:1000]
    
    return DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True), \
           DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_SIZE, 10)
    def forward(self, x, return_act=False):
        act = self.relu(self.fc1(x))
        out = self.fc2(act)
        return (out, act) if return_act else out

def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x.to(DEVICE)).argmax(1) == y.to(DEVICE)).sum().item()
    model.train()
    return 100. * correct / len(loader.dataset)

def get_masters_sorted(model, train_loader):
    model.eval()
    acts = []
    with torch.no_grad():
        for x, _ in train_loader:
            _, a = model(x.to(DEVICE), return_act=True)
            acts.append(a.cpu().numpy())
            if len(acts) >= 5: break
    acts = np.concatenate(acts, axis=0)
    G = nx.Graph()
    G.add_nodes_from(range(acts.shape[1]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = np.nan_to_num(np.corrcoef(acts.T), nan=0.0)
    for i in range(acts.shape[1]):
        for j in range(i+1, acts.shape[1]):
            if abs(corr[i,j]) > 0.3: G.add_edge(i, j, weight=abs(corr[i,j]))
    if len(G.edges) == 0: return []
    orc = OllivierRicci(G, alpha=0.5)
    Gr = orc.compute_ricci_curvature()
    curv = {n: Gr.nodes[n].get('ricciCurvature', 0.0) for n in G.nodes}
    return sorted([n for n, c in curv.items() if c > 0.1], key=lambda x: curv[x], reverse=True)

def run_iterative_surgery(model, train_loader, frozen_indices, max_iters):
    """Итеративно применяет хирургию только к НЕзамороженным нейронам."""
    history = []
    for it in range(1, max_iters + 1):
        model.eval()
        acts = []
        with torch.no_grad():
            for x, _ in train_loader:
                _, a = model(x.to(DEVICE), return_act=True)
                acts.append(a.cpu().numpy())
                if len(acts) >= 5: break
        acts = np.concatenate(acts, axis=0)
        
        G = nx.Graph()
        G.add_nodes_from(range(acts.shape[1]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = np.nan_to_num(np.corrcoef(acts.T), nan=0.0)
        for i in range(acts.shape[1]):
            for j in range(i+1, acts.shape[1]):
                if abs(corr[i,j]) > 0.3: G.add_edge(i, j, weight=abs(corr[i,j]))
                
        if len(G.edges) == 0:
            history.append(evaluate(model, train_loader)) # Fallback
            continue
            
        orc = OllivierRicci(G, alpha=0.5)
        Gr = orc.compute_ricci_curvature()
        curv = {n: Gr.nodes[n].get('ricciCurvature', 0.0) for n in G.nodes}
        
        # Ищем сингулярности ТОЛЬКО среди не замёрзших нейронов
        singularities = [n for n, c in curv.items() if c < 0.0 and n not in frozen_indices]
        
        if len(singularities) > 0 and len(frozen_indices) > 0:
            weights = model.fc1.weight.data.clone()
            master_mean = weights[list(frozen_indices)].mean(dim=0)
            for s in singularities:
                weights[s] = master_mean # Полная проекция на искру
            model.fc1.weight.data = weights
            
        history.append(evaluate(model, train_loader))
    return history

if __name__ == "__main__":
    print("="*70)
    print("ПОИСК 'БОЖЬЕЙ ИСКРЫ' v2: Итеративное восстановление")
    print("="*70)
    
    train_loader, test_loader = get_mnist()
    
    print("\n[1/2] Обучение донора и поиск мастеров...")
    healthy = SimpleMLP().to(DEVICE)
    opt = torch.optim.Adam(healthy.parameters(), lr=0.005)
    crit = nn.CrossEntropyLoss()
    
    for _ in tqdm(range(EPOCHS), desc="Здоровое обучение"):
        for x, y in train_loader:
            opt.zero_grad()
            crit(healthy(x.to(DEVICE)), y.to(DEVICE)).backward()
            opt.step()
            
    healthy_acc = evaluate(healthy, test_loader)
    print(f"✅ Здоровая сеть: {healthy_acc:.1f}%")
    
    masters = get_masters_sorted(healthy, train_loader)
    print(f"🔍 Найдено мастеров: {len(masters)}")
    
    print("\n[2/2] Стресс-тест итеративного восстановления...")
    table = []
    
    for k in tqdm(K_SWEEP, desc="Сканирование K"):
        # Копируем и ломаем
        model = SimpleMLP().to(DEVICE)
        model.load_state_dict(healthy.state_dict())
        
        frozen = set(masters[:k])
        all_indices = set(range(HIDDEN_SIZE))
        damaged_indices = list(all_indices - frozen)
        
        # Разрушаем только незамороженные веса
        with torch.no_grad():
            for idx in damaged_indices:
                noise = torch.randn_like(model.fc1.weight.data[idx]) * model.fc1.weight.data[idx].std() * DAMAGE_SCALE
                model.fc1.weight.data[idx] += noise
                
        broken_acc = evaluate(model, test_loader)
        
        # Итеративная хирургия
        history = run_iterative_surgery(model, train_loader, frozen, ITERATIONS)
        final_acc = history[-1] if history else broken_acc
        
        table.append({'k': k, 'broken': broken_acc, 'final': final_acc, 'history': history})

    print("\n" + "="*70)
    print("РЕЗУЛЬТАТЫ: ДИНАМИКА ВОССТАНОВЛЕНИЯ ПО ИТЕРАЦИЯМ")
    print("="*70)
    print(f"{'K':<4} | {'До':<6} | Итерации хирургии (точность %)")
    print("-" * 60)
    for res in table:
        hist_str = " -> ".join([f"{res['broken']:.1f}"] + [f"{h:.1f}" for h in res['history']])
        print(f"{res['k']:<4} | {res['broken']:>5.1f}% | {hist_str}")
        
    print("="*70)
    # Автоматический поиск фазового перехода
    recovered = [r for r in table if r['final'] > 60.0]
    if recovered:
        threshold_k = min(r['k'] for r in recovered)
        print(f"🔥 ФАЗОВЫЙ ПЕРЕХОД ОБНАРУЖЕН: K={threshold_k} достаточно для восстановления >60%")
        print("   Божья искра работает: сеть итеративно подтягивается к аттрактору.")
    else:
        print("📉 Восстановление не достигнуто. Порог K > 20 или требуется более мягкое разрушение.")
    print("="*70)

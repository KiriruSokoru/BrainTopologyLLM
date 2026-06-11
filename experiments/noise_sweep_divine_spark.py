"""
Эксперимент: Поиск "Божьей Искры" через Sweep по уровню шума.
Определяем критический порог разрушения, после которого сеть теряет способность к восстановлению.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import warnings
from tqdm import tqdm
import copy

# ========== НАСТРОЙКИ ==========
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_SIZE = 128
EPOCHS_HEALTHY = 8
EPOCHS_RECOVERY = 2  # Даем сети всего 2 эпохи на восстановление
NOISE_SWEEP = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]  # Уровень разрушения весов

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

def analyze_topology(model, train_loader):
    """Возвращает количество мастеров и сингулярностей."""
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
            if abs(corr[i,j]) > 0.3: 
                G.add_edge(i, j, weight=abs(corr[i,j]))
                
    if len(G.edges) == 0:
        return 0, 0
        
    orc = OllivierRicci(G, alpha=0.5)
    Gr = orc.compute_ricci_curvature()
    curv = {n: Gr.nodes[n].get('ricciCurvature', 0.0) for n in G.nodes}
    
    masters = sum(1 for c in curv.values() if c > 0.1)
    singularities = sum(1 for c in curv.values() if c < 0.0)
    return masters, singularities

def damage_model(model, ratio):
    """Разрушает указанную долю весов сильным шумом."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and len(param.shape) == 2:
                mask = torch.rand_like(param) < ratio
                # Генерируем шум того же размера, что и param
                noise = torch.randn_like(param) * param.std() * 5.0
                # ИСПРАВЛЕНО: применяем маску и к шуму тоже!
                param[mask] += noise[mask]

def train_model(model, train_loader, epochs):
    opt = optim.Adam(model.parameters(), lr=0.005)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in train_loader:
            opt.zero_grad()
            crit(model(x.to(DEVICE)), y.to(DEVICE)).backward()
            opt.step()

# ========== ГЛАВНЫЙ ПРОЦЕСС ==========
if __name__ == "__main__":
    print("="*80)
    print("ПОИСК 'БОЖЬЕЙ ИСКРЫ': Sweep по уровню разрушения")
    print("="*80)
    
    train_loader, test_loader = get_mnist()
    
    # 1. Обучаем эталонную здоровую сеть
    print("\n[1/3] Обучение эталонной здоровой сети...")
    healthy_model = SimpleMLP().to(DEVICE)
    train_model(healthy_model, train_loader, EPOCHS_HEALTHY)
    healthy_acc = evaluate(healthy_model, test_loader)
    healthy_masters, healthy_sing = analyze_topology(healthy_model, train_loader)
    
    print(f"✅ Эталон: Точность={healthy_acc:.1f}%, Мастеров={healthy_masters}, Сингулярностей={healthy_sing}")
    
    # 2. Sweep по шуму
    print("\n[2/3] Стресс-тест: увеличение уровня разрушения...")
    results = []
    
    for noise_level in tqdm(NOISE_SWEEP, desc="Уровень шума"):
        # Берем свежую копию эталона
        model = copy.deepcopy(healthy_model)
        
        # Ломаем
        damage_model(model, noise_level)
        broken_acc = evaluate(model, test_loader)
        broken_m, broken_s = analyze_topology(model, train_loader)
        
        # Пытаемся восстановить (даем 2 эпохи)
        train_model(model, train_loader, EPOCHS_RECOVERY)
        recovered_acc = evaluate(model, test_loader)
        recovered_m, recovered_s = analyze_topology(model, train_loader)
        
        results.append({
            'noise': noise_level,
            'broken_acc': broken_acc,
            'recovered_acc': recovered_acc,
            'broken_m': broken_m,
            'recovered_m': recovered_m,
            'recovery_delta': recovered_acc - broken_acc
        })

    # 3. Вывод результатов
    print("\n" + "="*80)
    print("ИТОГОВАЯ ТАБЛИЦА: ГРАНИЦА ЖИЗНИ И СМЕРТИ ТОПОЛОГИИ")
    print("="*80)
    print(f"{'Шум':<6} | {'До починки':<10} | {'Мастеров':<10} | {'После 2 эпох':<12} | {'Мастеров':<10} | {'Δ Точности'}")
    print("-" * 80)
    
    spark_found = False
    for res in results:
        # "Божья искра" найдена, если сеть была почти мертва (<30%), но восстановилась (>70%), 
        # и при этом в ней остался хотя бы 1 мастер.
        is_spark = res['broken_acc'] < 40.0 and res['recovered_acc'] > 75.0 and res['recovered_m'] >= 1
        
        marker = " 🔥 БОЖЬЯ ИСКРА!" if is_spark else ""
        if is_spark and not spark_found:
            spark_found = True
            
        print(f"{res['noise']*100:>3.0f}%   | "
              f"{res['broken_acc']:>5.1f}%      | "
              f"{res['broken_m']:>4}       | "
              f"{res['recovered_acc']:>5.1f}%        | "
              f"{res['recovered_m']:>4}       | "
              f"+{res['recovery_delta']:>5.1f} п.п. {marker}")
              
    print("="*80)
    if spark_found:
        print("🎯 УСПЕХ: Мы нашли минимальный инвариант! Даже при сильном разрушении")
        print("   сеть сохраняет топологическое 'ядро' (мастеров), способное запустить")
        print("   восстановление через градиентный спуск.")
    else:
        print("📉 НАБЛЮДЕНИЕ: Линейная деградация. Топологическое ядро разрушается")
        print("   пропорционально шуму. Попробуем увеличить HIDDEN_SIZE или изменить порог графа.")
    print("="*80)

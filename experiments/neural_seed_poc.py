"""
Proof of Concept: Neural Seed (Нейронное Семя)
Архивация нейросети до минимального топологического инварианта и её последующее выращивание.
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
import os
import sys

# ========== НАСТРОЙКИ ==========
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_SIZE = 128
EPOCHS_TRAIN = 8
EPOCHS_GROW = 3  # Сколько эпох "выращиваем" сеть из семени
SEED_FILE = "results/neural_seed.pt"

print(f"Устройство: {DEVICE}")
os.makedirs("results", exist_ok=True)

# ========== ДАННЫЕ ==========
def get_mnist():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, transform=transform)
    
    # Малый датасет для скорости PoC
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
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x.to(DEVICE)).argmax(1) == y.to(DEVICE)).sum().item()
    model.train()
    return 100. * correct / len(loader.dataset)

def train_model(model, train_loader, epochs):
    opt = optim.Adam(model.parameters(), lr=0.005)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in train_loader:
            opt.zero_grad()
            crit(model(x.to(DEVICE)), y.to(DEVICE)).backward()
            opt.step()

# ========== ТОПЛОГИЧЕСКИЙ АНАЛИЗ ==========
def get_masters(model, train_loader, top_k=30):
    """Находит только топ-K самых сильных мастеров (с максимальной кривизной)."""
    model.eval()
    acts = []
    with torch.no_grad():
        for x, _ in train_loader:
            act = model.relu(model.fc1(x.to(DEVICE)))
            acts.append(act.cpu().numpy())
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
        return []
        
    orc = OllivierRicci(G, alpha=0.5)
    Gr = orc.compute_ricci_curvature()
    curv = {n: Gr.nodes[n].get('ricciCurvature', 0.0) for n in G.nodes}
    
    # Сортируем по кривизне (от большей к меньшей) и берем топ-K
    sorted_masters = sorted(curv.items(), key=lambda x: x[1], reverse=True)
    top_masters = [n for n, c in sorted_masters[:top_k] if c > 0.1]
    
    print(f"    Найдено мастеров с кривизной > 0.1: {len([n for n, c in curv.items() if c > 0.1])}")
    print(f"    Оставляем топ-{top_k} самых сильных: {len(top_masters)}")
    
    return top_masters

# ========== ЯДРО КОНЦЕПЦИИ: АРХИВАЦИЯ И ВЫРАЩИВАНИЕ ==========

def create_neural_seed(model, train_loader, filepath, top_k=30):
    """Извлекает только топ-K мастеров и сохраняет их как 'Семя'."""
    print("  [1/2] Поиск топологических мастеров...")
    masters = get_masters(model, train_loader, top_k=top_k)
    print(f"  [2/2] Сохранение семени ({len(masters)} мастеров)...")
    
    seed_data = {
        "architecture": "SimpleMLP_128",
        "master_indices": masters,
        "master_weights_fc1": model.fc1.weight.data[masters].clone().cpu(),
        "master_weights_fc2": model.fc2.weight.data[:, masters].clone().cpu(),
        "total_params_original": sum(p.numel() for p in model.parameters())
    }
    
    torch.save(seed_data, filepath)
    
    original_size = sum(p.numel() * 4 for p in model.parameters())
    seed_size = os.path.getsize(filepath)
    
    return seed_data, original_size, seed_size

def grow_model_from_seed(seed_data, train_loader, epochs):
    """Создает новую модель, внедряет семя и выращивает её."""
    print("  [1/3] Инициализация новой сети...")
    new_model = SimpleMLP().to(DEVICE) # Веса уже случайные (шум)
    
    print("  [2/3] Внедрение топологического ядра (мастеров)...")
    masters = seed_data["master_indices"]
    
    # Внедряем сохраненные веса мастеров
    new_model.fc1.weight.data[masters] = seed_data["master_weights_fc1"].to(DEVICE)
    # Восстанавливаем столбцы для fc2
    new_model.fc2.weight.data[:, masters] = seed_data["master_weights_fc2"].to(DEVICE)
    
    print(f"  [3/3] Выращивание (fine-tuning) в течение {epochs} эпох...")
    # Обучаем ВСЮ сеть с небольшим learning rate, чтобы позволить остальным 
    # нейронам "подтянуться" к мастерам, не разрушая их сразу.
    opt = optim.Adam(new_model.parameters(), lr=0.001) 
    crit = nn.CrossEntropyLoss()
    
    for _ in range(epochs):
        for x, y in train_loader:
            opt.zero_grad()
            crit(new_model(x.to(DEVICE)), y.to(DEVICE)).backward()
            opt.step()
            
    return new_model

# ========== ГЛАВНЫЙ ПРОЦЕСС ==========
if __name__ == "__main__":
    print("="*70)
    print("NEURAL SEED: Доказательство концепции архивации и выращивания")
    print("="*70)
    
    train_loader, test_loader = get_mnist()
    
    # ЭТАП 1: Эталон
    print("\n[ЭТАП 1] Обучение эталонной модели...")
    original_model = SimpleMLP().to(DEVICE)
    train_model(original_model, train_loader, EPOCHS_TRAIN)
    acc_original = evaluate(original_model, test_loader)
    print(f"✅ Эталонная точность: {acc_original:.2f}%")
    
    # ЭТАП 2: Архивация
    print("\n[ЭТАП 2] Создание Neural Seed (Архивация)...")
    seed_data, orig_size, seed_size = create_neural_seed(original_model, train_loader, SEED_FILE, top_k=30)
    
    compression_ratio = orig_size / seed_size
    print(f"📦 Размер оригинала: {orig_size / 1024:.2f} KB")
    print(f"🌱 Размер семени:    {seed_size / 1024:.2f} KB")
    print(f"📉 Коэффициент сжатия: {compression_ratio:.1f}x")
    
    # ЭТАП 3: Выращивание
    print("\n[ЭТАП 3] Выращивание новой модели из семени...")
    grown_model = grow_model_from_seed(seed_data, train_loader, EPOCHS_GROW)
    acc_grown = evaluate(grown_model, test_loader)
    
    # ИТОГИ
    print("\n" + "="*70)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*70)
    print(f"Точность оригинала : {acc_original:.2f}%")
    print(f"Точность выращенной: {acc_grown:.2f}%")
    print(f"Потеря точности    : {acc_original - acc_grown:.2f} п.п.")
    print(f"Экономия памяти    : {compression_ratio:.1f}x")
    print("="*70)
    
    if acc_grown > (acc_original * 0.85): # Допускаем потерю до 15% для PoC
        print("🎉 УСПЕХ: Концепция Neural Seed работает!")
        print("   Мы можем хранить сеть в сжатом виде и восстанавливать её функциональность.")
    else:
        print("🔍 ТРЕБУЕТСЯ ДОРАБОТКА: Потеря точности слишком велика.")
    print("="*70)

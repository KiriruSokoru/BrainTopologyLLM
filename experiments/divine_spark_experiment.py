"""
Эксперимент: Поиск "Божьей Искры" (Минимальный Инвариант).
Сколько мастеров достаточно, чтобы вытащить разрушенную сеть из хаоса?
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

# ========== НАСТРОЙКИ ==========
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_SIZE = 128
EPOCHS = 10
DAMAGE_RATIO = 0.4  # Уничтожаем 40% весов, создавая хаос
MASTER_SWEEP = [1, 2, 3, 5, 10, 20, 50]  # Проверяем разное количество "искр"

print(f"Устройство: {DEVICE}")

# ========== ДАННЫЕ ==========
def get_mnist():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, transform=transform)
    
    # Берем подмножество для скорости
    X_train, y_train = train_ds.data[:3000].float().view(-1, 784) / 255.0, train_ds.targets[:3000]
    X_test, y_test = test_ds.data[:1000].float().view(-1, 784) / 255.0, test_ds.targets[:1000]
    
    return DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True), \
           DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

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
        if return_act: return out, act
        return out

# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x.to(DEVICE))
            correct += (out.argmax(1) == y.to(DEVICE)).sum().item()
    return 100. * correct / len(test_loader.dataset)

def damage_model(model):
    """Жестко портим веса, создавая сингулярности."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and len(param.shape) == 2:
                noise = torch.randn_like(param) * param.std() * 10.0 # Сильный шум
                mask = torch.rand_like(param) < DAMAGE_RATIO
                param[mask] += noise[mask]

def get_true_masters(model, train_loader):
    """Находим настоящих мастеров в здоровой сети."""
    model.eval()
    acts = []
    with torch.no_grad():
        for x, _ in train_loader:
            _, act = model(x.to(DEVICE), return_act=True)
            acts.append(act.cpu().numpy())
            if len(acts) >= 5: break
            
    acts = np.concatenate(acts, axis=0)
    G = nx.Graph()
    G.add_nodes_from(range(acts.shape[1]))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = np.nan_to_num(np.corrcoef(acts.T), nan=0.0)
        
    for i in range(acts.shape[1]):
        for j in range(i + 1, acts.shape[1]):
            if abs(corr[i, j]) > 0.3:
                G.add_edge(i, j, weight=abs(corr[i, j]))
                
    orc = OllivierRicci(G, alpha=0.5)
    G_ricci = orc.compute_ricci_curvature()
    
    curvatures = {node: G_ricci.nodes[node].get('ricciCurvature', 0.0) for node in G.nodes}
    # Сортируем мастеров по убыванию кривизны (самые стабильные первые)
    masters = sorted([node for node, c in curvatures.items() if c > 0.1], key=lambda x: curvatures[x], reverse=True)
    return masters

def apply_restricted_surgery(model, train_loader, true_masters, k_masters):
    """Применяем хирургию, используя ТОЛЬКО top-K мастеров."""
    model.eval()
    acts = []
    with torch.no_grad():
        for x, _ in train_loader:
            _, act = model(x.to(DEVICE), return_act=True)
            acts.append(act.cpu().numpy())
            if len(acts) >= 5: break
            
    acts = np.concatenate(acts, axis=0)
    G = nx.Graph()
    G.add_nodes_from(range(acts.shape[1]))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = np.nan_to_num(np.corrcoef(acts.T), nan=0.0)
        
    for i in range(acts.shape[1]):
        for j in range(i + 1, acts.shape[1]):
            if abs(corr[i, j]) > 0.3:
                G.add_edge(i, j, weight=abs(corr[i, j]))
                
    if len(G.edges) == 0:
        model.train()
        return 0
        
    orc = OllivierRicci(G, alpha=0.5)
    G_ricci = orc.compute_ricci_curvature()
    
    curvatures = {node: G_ricci.nodes[node].get('ricciCurvature', 0.0) for node in G.nodes}
    singularities = [node for node, c in curvatures.items() if c < 0.0]
    
    # БЕРЕМ ТОЛЬКО K МАСТЕРОВ
    allowed_masters = true_masters[:k_masters]
    
    if len(singularities) > 0 and len(allowed_masters) > 0:
        weights = model.fc1.weight.data.clone()
        master_weights = weights[allowed_masters].mean(dim=0)
        
        for sing in singularities:
            weights[sing] = master_weights # Полная замена для максимального эффекта спасения
            
        model.fc1.weight.data = weights
        
    model.train()
    return len(singularities)

# ========== ГЛАВНЫЙ ПРОЦЕСС ==========
if __name__ == "__main__":
    print("="*70)
    print("ЭКСПЕРИМЕНТ: ПОИСК 'БОЖЬЕЙ ИСКРЫ' (Минимальный Инвариант)")
    print(f"Условия: Разрушение {DAMAGE_RATIO*100:.0f}% весов, восстановление через K мастеров")
    print("="*70)
    
    train_loader, test_loader = get_mnist()
    
    # 1. Обучаем здоровую сеть-донор
    print("\n[1/4] Обучение здоровой сети-донора...")
    healthy_model = SimpleMLP().to(DEVICE)
    opt = optim.Adam(healthy_model.parameters(), lr=0.005)
    crit = nn.CrossEntropyLoss()
    
    for epoch in tqdm(range(EPOCHS), desc="Здоровое обучение"):
        for x, y in train_loader:
            opt.zero_grad()
            loss = crit(healthy_model(x.to(DEVICE)), y.to(DEVICE))
            loss.backward()
            opt.step()
            
    healthy_acc = evaluate(healthy_model, test_loader)
    print(f"✅ Здоровая сеть: {healthy_acc:.2f}% точности")
    
    true_masters = get_true_masters(healthy_model, train_loader)
    print(f"🔍 Найдено истинных мастеров в здоровой сети: {len(true_masters)}")
    
    # 2. Sweep по количеству мастеров
    print("\n[2/4] Начало стресс-теста восстановления...")
    results = []
    
    for k in tqdm(MASTER_SWEEP, desc="Тестирование K мастеров"):
        # Берем копию здоровой модели
        model = SimpleMLP().to(DEVICE)
        model.load_state_dict(healthy_model.state_dict())
        
        # Ломаем её
        damage_model(model)
        broken_acc = evaluate(model, test_loader)
        
        # Чиним, используя только K мастеров
        suppressed = apply_restricted_surgery(model, train_loader, true_masters, k)
        fixed_acc = evaluate(model, test_loader)
        
        results.append({
            'k': k,
            'broken_acc': broken_acc,
            'fixed_acc': fixed_acc,
            'suppressed': suppressed,
            'recovery': fixed_acc - broken_acc
        })
        
    # 3. Вывод результатов
    print("\n" + "="*70)
    print("ИТОГОВАЯ ТАБЛИЦА: ЭФФЕКТ 'БОЖЬЕЙ ИСКРЫ'")
    print("="*70)
    print(f"{'K мастеров':<12} | {'До хирургии':<12} | {'После':<12} | {'Восстановлено':<15} | {'Подавлено синг.'}")
    print("-" * 70)
    
    phase_transition_found = False
    for res in results:
        marker = " <-- ФАЗОВЫЙ ПЕРЕХОД!" if res['recovery'] > 15.0 and not phase_transition_found else ""
        if res['recovery'] > 15.0 and not phase_transition_found:
            phase_transition_found = True
            
        print(f"{res['k']:<12} | {res['broken_acc']:>5.2f}%       | {res['fixed_acc']:>5.2f}%       | +{res['recovery']:>6.2f} п.п.      | {res['suppressed']}")
        
    print("="*70)
    if phase_transition_found:
        print("🔥 НАЙДЕНО: Существует четкий порог (Божья искра), после которого сеть")
        print("   магическим образом восстанавливается из хаоса!")
    else:
        print("📈 НАБЛЮДЕНИЕ: Восстановление линейно зависит от количества мастеров.")
    print("="*70)

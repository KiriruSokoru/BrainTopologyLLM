import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings
import os
from tqdm import tqdm # Добавляем прогресс-бар

# ========== НАСТРОЙКИ ==========
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS_ORACLE = 2      
EPOCHS_GROW = 1        
TOP_K_VALUES = [50, 20, 5] 

print(f"Устройство: {DEVICE}")
os.makedirs("results", exist_ok=True)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# ========== ДАННЫЕ ==========
def get_json_dataset(tokenizer, num_samples=500):
    texts = []
    for i in range(num_samples):
        texts.append(f'{{"id": {i}, "status": "ok", "code": 200}}')
    
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
    
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        def __len__(self):
            return len(self.encodings['input_ids'])
        def __getitem__(self, idx):
            return {
                'input_ids': self.encodings['input_ids'][idx],
                'attention_mask': self.encodings['attention_mask'][idx],
                'labels': self.encodings['input_ids'][idx].clone()
            }
    return SimpleDataset(encodings)

# ========== ТОПЛОГИЧЕСКИЙ АНАЛИЗ (РАДИКАЛЬНО ОПТИМИЗИРОВАННЫЙ) ==========
def get_llm_masters(model, dataloader, top_k=50):
    print("  [1/3] Сбор активаций последнего MLP слоя...")
    model.eval()
    acts = []
    
    def hook_fn(module, input, output):
        acts.append(output.mean(dim=1).detach().cpu().numpy())

    last_mlp_fc = model.transformer.h[-1].mlp.c_fc
    handle = last_mlp_fc.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for batch in dataloader:
            model(input_ids=batch['input_ids'].to(DEVICE), attention_mask=batch['attention_mask'].to(DEVICE))
            if len(acts) >= 10: 
                break
                
    handle.remove()
    
    acts = np.concatenate(acts, axis=0)
    print(f"    Размер матрицы активаций: {acts.shape}")
    
    print("  [2/3] Построение разреженного графа корреляций (Degree Capping)...")
    G = nx.Graph()
    num_nodes = acts.shape[1]
    G.add_nodes_from(range(num_nodes))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = np.nan_to_num(np.abs(np.corrcoef(acts.T)), nan=0.0)
    
    CORR_THRESHOLD = 0.4
    MAX_DEGREE = 20 # Ключевая оптимизация: макс. 20 связей на узел
    capped_edges = []
    
    # Быстрый поиск топ-связей для каждого узла
    for i in range(num_nodes):
        # Смотрим только на узлы с большим индексом, чтобы избежать дубликатов ребер
        row_corr = corr[i, i+1:]
        if len(row_corr) == 0:
            continue
            
        # Находим индексы топ-MAX_DEGREE самых сильных связей
        top_local_indices = np.argsort(row_corr)[-MAX_DEGREE:]
        
        for idx in top_local_indices:
            j = i + 1 + idx
            weight = row_corr[idx]
            if weight > CORR_THRESHOLD:
                capped_edges.append((i, j, weight))
                
    G.add_weighted_edges_from(capped_edges)
    print(f"    ✅ Граф построен: {num_nodes} узлов, {len(capped_edges)} ребер (после ограничения степени)")
    
    if len(capped_edges) == 0:
        print("    ⚠️ Граф пуст. Понижаем порог до 0.2...")
        CORR_THRESHOLD = 0.2
        capped_edges = []
        for i in range(num_nodes):
            row_corr = corr[i, i+1:]
            if len(row_corr) == 0: continue
            top_local_indices = np.argsort(row_corr)[-MAX_DEGREE:]
            for idx in top_local_indices:
                j = i + 1 + idx
                weight = row_corr[idx]
                if weight > CORR_THRESHOLD:
                    capped_edges.append((i, j, weight))
        G.add_weighted_edges_from(capped_edges)
        print(f"    ✅ Граф перестроен: {len(capped_edges)} ребер")

    print("  [3/3] Расчет кривизны Риччи (теперь это займет ~10-15 сек на CPU)...")
    orc = OllivierRicci(G, alpha=0.5)
    
    # Расчет кривизны
    Gr = orc.compute_ricci_curvature()
    curv = {n: Gr.nodes[n].get('ricciCurvature', 0.0) for n in G.nodes}
    
    sorted_masters = sorted(curv.items(), key=lambda x: x[1], reverse=True)
    top_masters = [n for n, c in sorted_masters[:top_k] if c > 0.01]
    
    print(f"    Найдено мастеров с кривизной > 0.01: {len([n for n, c in curv.items() if c > 0.01])}")
    print(f"    Оставляем топ-{top_k}: {len(top_masters)}")
    
    return top_masters

def create_llm_seed(model, dataloader, top_k=50):
    masters = get_llm_masters(model, dataloader, top_k=top_k)
    
    last_layer_idx = len(model.transformer.h) - 1
    mlp_fc1 = model.transformer.h[last_layer_idx].mlp.c_fc   
    mlp_fc2 = model.transformer.h[last_layer_idx].mlp.c_proj 
    
    seed_data = {
        "architecture": "distilgpt2_last_mlp",
        "master_indices": masters,
        "master_weights_fc1": mlp_fc1.weight.data[:, masters].clone().cpu(), 
        "master_weights_fc2": mlp_fc2.weight.data[masters, :].clone().cpu(), 
        "master_bias_fc1": mlp_fc1.bias.data[masters].clone().cpu(),         
        "total_params_original": sum(p.numel() for p in model.parameters())
    }
    
    return seed_data

def evaluate_perplexity(model, dataloader):
    model.eval()
    total_loss = 0
    total_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                input_ids=batch['input_ids'].to(DEVICE),
                attention_mask=batch['attention_mask'].to(DEVICE),
                labels=batch['labels'].to(DEVICE)
            )
            total_loss += outputs.loss.item()
            total_batches += 1
    
    avg_loss = total_loss / total_batches
    perplexity = np.exp(avg_loss)
    model.train()
    return perplexity

def train_last_mlp_only(model, dataloader, epochs, lr=5e-4):
    for name, param in model.named_parameters():
        param.requires_grad = False
        
    last_layer_idx = len(model.transformer.h) - 1
    model.transformer.h[last_layer_idx].mlp.c_fc.weight.requires_grad = True
    model.transformer.h[last_layer_idx].mlp.c_proj.weight.requires_grad = True
    model.transformer.h[last_layer_idx].mlp.c_fc.bias.requires_grad = True

    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    for _ in range(epochs):
        for batch in dataloader:
            opt.zero_grad()
            outputs = model(
                input_ids=batch['input_ids'].to(DEVICE),
                attention_mask=batch['attention_mask'].to(DEVICE),
                labels=batch['labels'].to(DEVICE)
            )
            outputs.loss.backward()
            opt.step()
    return model

# ========== ГЛАВНЫЙ ПРОЦЕСС ==========
if __name__ == "__main__":
    print("="*70)
    print("NEURAL SEED: LLM PoC (ОПТИМИЗИРОВАННЫЙ)")
    print("="*70)
    
    print("Загрузка токенизатора и модели...")
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = get_json_dataset(tokenizer, num_samples=500)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    
    print("\n[ЭТАП 1] Обучение ЭТАЛОНА (2 эпохи)...")
    set_seed(42)
    oracle_model = GPT2LMHeadModel.from_pretrained('distilgpt2').to(DEVICE)
    oracle_model = train_last_mlp_only(oracle_model, train_loader, epochs=EPOCHS_ORACLE)
    ppl_oracle = evaluate_perplexity(oracle_model, train_loader)
    print(f"✅ Перплексия Эталона: {ppl_oracle:.2f}")
    
    print("\n[ЭТАП 2] Извлечение мастеров...")
    seeds = {}
    orig_size_kb = sum(p.numel() for p in oracle_model.parameters()) * 4 / 1024
    
    for k in TOP_K_VALUES:
        print(f"\n--- Генерация семени для top_k={k} ---")
        seeds[k] = create_llm_seed(oracle_model, train_loader, top_k=k)
        
        seed_size_bytes = (
            seeds[k]["master_weights_fc1"].numel() * 4 +
            seeds[k]["master_weights_fc2"].numel() * 4 +
            seeds[k]["master_bias_fc1"].numel() * 4
        )
        seed_size_kb = seed_size_bytes / 1024
        print(f"📦 Оригинал: {orig_size_kb:.1f} KB | 🌱 Семя: {seed_size_kb:.2f} KB | Сжатие: {orig_size_kb/seed_size_kb:.1f}x")

    print("\n[ЭТАП 3] Выращивание моделей (1 эпоха)...")
    results = {}
    
    print("  [A] Baseline (шум, 1 эпоха)...")
    set_seed(42)
    baseline_model = GPT2LMHeadModel.from_pretrained('distilgpt2').to(DEVICE)
    nn.init.normal_(baseline_model.transformer.h[-1].mlp.c_fc.weight, std=0.02)
    nn.init.normal_(baseline_model.transformer.h[-1].mlp.c_proj.weight, std=0.02)
    baseline_model = train_last_mlp_only(baseline_model, train_loader, epochs=EPOCHS_GROW)
    results['baseline'] = evaluate_perplexity(baseline_model, train_loader)
    print(f"      -> Perplexity: {results['baseline']:.2f}")

    for k in TOP_K_VALUES:
        print(f"  [B] Seed (top_k={k}, 1 эпоха)...")
        set_seed(42)
        grown_model = GPT2LMHeadModel.from_pretrained('distilgpt2').to(DEVICE)
        
        masters = seeds[k]["master_indices"]
        last_layer_idx = len(grown_model.transformer.h) - 1
        grown_model.transformer.h[last_layer_idx].mlp.c_fc.weight.data[:, masters] = seeds[k]["master_weights_fc1"].to(DEVICE)
        grown_model.transformer.h[last_layer_idx].mlp.c_proj.weight.data[masters, :] = seeds[k]["master_weights_fc2"].to(DEVICE)
        grown_model.transformer.h[last_layer_idx].mlp.c_fc.bias.data[masters] = seeds[k]["master_bias_fc1"].to(DEVICE)
        
        grown_model = train_last_mlp_only(grown_model, train_loader, epochs=EPOCHS_GROW)
        results[k] = evaluate_perplexity(grown_model, train_loader)
        print(f"      -> Perplexity: {results[k]:.2f}")

    print("\n" + "="*70)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ: LLM NEURAL SEED")
    print("="*70)
    print(f"{'Конфигурация':<25} | {'Perplexity (1 эпоха)':<20} | {'Отставание от Эталона':<20}")
    print("-" * 70)
    
    for k in ['baseline'] + TOP_K_VALUES:
        ppl = results[k]
        diff = ppl - ppl_oracle
        label = "Baseline (шум)" if k == 'baseline' else f"Seed (top_k={k})"
        print(f"{label:<25} | {ppl:<20.2f} | {diff:+.2f}")
        
    print("="*70)

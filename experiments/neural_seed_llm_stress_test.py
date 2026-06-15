import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings
import os
import random

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS_ORACLE = 3      # Увеличиваем, так как задача сложнее
EPOCHS_GROW = 1        
# Расширяем диапазон, чтобы поймать переход
TOP_K_VALUES = [100, 50, 20, 5] 

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# ========== СЛОЖНЫЙ ДАТАСЕТ (Высокая внутренняя размерность) ==========
def get_complex_dataset(tokenizer, num_samples=800):
    texts = []
    for _ in range(num_samples):
        task_type = random.choice(['math', 'logic', 'format', 'cipher'])
        if task_type == 'math':
            a, b = random.randint(1, 50), random.randint(1, 50)
            texts.append(f'Q: {a} + {b} = ?\nA: {a+b}')
        elif task_type == 'logic':
            val = random.choice(['true', 'false'])
            texts.append(f'Q: Is 5 > 3?\nA: {val}') 
        elif task_type == 'format':
            d, m, y = random.randint(1,28), random.randint(1,12), random.randint(2020,2025)
            texts.append(f'Q: Date {y}-{m:02d}-{d:02d} to EU\nA: {d:02d}/{m:02d}/{y}')
        else: # cipher (Caesar +1)
            word = random.choice(['cat', 'dog', 'sun', 'sky', 'box'])
            ciphered = "".join(chr(ord(c)+1) for c in word)
            texts.append(f'Q: Encode {word}\nA: {ciphered}')
    
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
    
    class ComplexDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        def __len__(self): return len(self.encodings['input_ids'])
        def __getitem__(self, idx):
            return {
                'input_ids': self.encodings['input_ids'][idx],
                'attention_mask': self.encodings['attention_mask'][idx],
                'labels': self.encodings['input_ids'][idx].clone()
            }
    return ComplexDataset(encodings)

# ========== ТОПЛОГИЧЕСКИЙ АНАЛИЗ (Оптимизированный) ==========
def get_llm_masters(model, dataloader, top_k=50):
    model.eval()
    acts = []
    def hook_fn(module, input, output):
        acts.append(output.mean(dim=1).detach().cpu().numpy())
    
    last_mlp_fc = model.transformer.h[-1].mlp.c_fc
    handle = last_mlp_fc.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for batch in dataloader:
            model(input_ids=batch['input_ids'].to(DEVICE), attention_mask=batch['attention_mask'].to(DEVICE))
            if len(acts) >= 15: break # Чуть больше данных для сложной задачи
    handle.remove()
    
    acts = np.concatenate(acts, axis=0)
    G = nx.Graph()
    num_nodes = acts.shape[1]
    G.add_nodes_from(range(num_nodes))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = np.nan_to_num(np.abs(np.corrcoef(acts.T)), nan=0.0)
    
    CORR_THRESHOLD = 0.35 # Чуть ниже, так как задачи разнообразнее
    MAX_DEGREE = 20
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
    print(f"    ✅ Граф: {num_nodes} узлов, {len(capped_edges)} ребер")
    
    orc = OllivierRicci(G, alpha=0.5)
    Gr = orc.compute_ricci_curvature()
    curv = {n: Gr.nodes[n].get('ricciCurvature', 0.0) for n in G.nodes}
    
    sorted_masters = sorted(curv.items(), key=lambda x: x[1], reverse=True)
    top_masters = [n for n, c in sorted_masters[:top_k] if c > -0.1] # Разрешаем чуть ниже 0 для разнообразия
    
    print(f"    ✅ Оставляем топ-{top_k}: {len(top_masters)} мастеров")
    return top_masters

def evaluate_perplexity(model, dataloader):
    model.eval()
    total_loss, total_batches = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(input_ids=batch['input_ids'].to(DEVICE), 
                           attention_mask=batch['attention_mask'].to(DEVICE),
                           labels=batch['labels'].to(DEVICE))
            total_loss += outputs.loss.item()
            total_batches += 1
    
    # ИСПРАВЛЕНО: total_batches вместо totalales
    return np.exp(total_loss / total_batches)

def train_last_mlp_only(model, dataloader, epochs, lr=5e-4):
    for param in model.parameters(): param.requires_grad = False
    last_idx = len(model.transformer.h) - 1
    model.transformer.h[last_idx].mlp.c_fc.weight.requires_grad = True
    model.transformer.h[last_idx].mlp.c_proj.weight.requires_grad = True
    model.transformer.h[last_idx].mlp.c_fc.bias.requires_grad = True

    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    for _ in range(epochs):
        for batch in dataloader:
            opt.zero_grad()
            outputs = model(input_ids=batch['input_ids'].to(DEVICE), labels=batch['labels'].to(DEVICE))
            outputs.loss.backward()
            opt.step()
    return model

if __name__ == "__main__":
    print("="*70)
    print("NEURAL SEED: СТРЕСС-ТЕСТ (Высокая внутренняя размерность)")
    print("="*70)
    
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token
    dataset = get_complex_dataset(tokenizer, num_samples=800)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    
    print("\n[1] Обучение Эталона (3 эпохи)...")
    set_seed(42)
    oracle = GPT2LMHeadModel.from_pretrained('distilgpt2').to(DEVICE)
    oracle = train_last_mlp_only(oracle, train_loader, epochs=EPOCHS_ORACLE)
    ppl_oracle = evaluate_perplexity(oracle, train_loader)
    print(f"✅ Перплексия Эталона: {ppl_oracle:.2f}")
    
    print("\n[2] Извлечение мастеров...")
    seeds = {}
    for k in TOP_K_VALUES:
        print(f"  -> top_k={k}")
        masters = get_llm_masters(oracle, train_loader, top_k=k)
        last_idx = len(oracle.transformer.h) - 1
        seeds[k] = {
            "masters": masters,
            "fc1_w": oracle.transformer.h[last_idx].mlp.c_fc.weight.data[:, masters].clone().cpu(),
            "fc2_w": oracle.transformer.h[last_idx].mlp.c_proj.weight.data[masters, :].clone().cpu(),
            "fc1_b": oracle.transformer.h[last_idx].mlp.c_fc.bias.data[masters].clone().cpu()
        }
        size_kb = (seeds[k]["fc1_w"].numel() * 4 + seeds[k]["fc2_w"].numel() * 4 + seeds[k]["fc1_b"].numel() * 4) / 1024
        print(f"     🌱 Размер семени: {size_kb:.1f} KB")

    print("\n[3] Выращивание (1 эпоха)...")
    results = {}
    
    # Baseline
    set_seed(42)
    baseline = GPT2LMHeadModel.from_pretrained('distilgpt2').to(DEVICE)
    nn.init.normal_(baseline.transformer.h[-1].mlp.c_fc.weight, std=0.02)
    baseline = train_last_mlp_only(baseline, train_loader, epochs=EPOCHS_GROW)
    results['baseline'] = evaluate_perplexity(baseline, train_loader)
    print(f"  Baseline (шум, 1 эпоха): {results['baseline']:.2f}")

    for k in TOP_K_VALUES:
        set_seed(42)
        grown = GPT2LMHeadModel.from_pretrained('distilgpt2').to(DEVICE)
        last_idx = len(grown.transformer.h) - 1
        m = seeds[k]["masters"]
        grown.transformer.h[last_idx].mlp.c_fc.weight.data[:, m] = seeds[k]["fc1_w"].to(DEVICE)
        grown.transformer.h[last_idx].mlp.c_proj.weight.data[m, :] = seeds[k]["fc2_w"].to(DEVICE)
        grown.transformer.h[last_idx].mlp.c_fc.bias.data[m] = seeds[k]["fc1_b"].to(DEVICE)
        
        grown = train_last_mlp_only(grown, train_loader, epochs=EPOCHS_GROW)
        results[k] = evaluate_perplexity(grown, train_loader)
        print(f"  Seed top_k={k:<3} (1 эпоха): {results[k]:.2f} (отставание: {results[k]-ppl_oracle:+.2f})")

    print("\n" + "="*70)
    print("ВЫВОД: Если top_k=5 показывает перплексию > 5.0, а top_k=50 держится около эталона,")
    print("мы официально зафиксировали фазовый переход деградации для сложных задач.")
    print("="*70)

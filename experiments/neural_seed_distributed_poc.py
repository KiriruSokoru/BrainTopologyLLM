import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS_ORACLE = 2
EPOCHS_GROW = 1
MASTERS_PER_LAYER = 10 # Берем по 10 мастеров с каждого из 6 слоев

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_json_dataset(tokenizer, num_samples=500):
    texts = [f'{{"id": {i}, "status": "ok", "code": 200}}' for i in range(num_samples)]
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, enc): self.enc = enc
        def __len__(self): return len(self.enc['input_ids'])
        def __getitem__(self, idx):
            return {'input_ids': self.enc['input_ids'][idx], 'attention_mask': self.enc['attention_mask'][idx], 'labels': self.enc['input_ids'][idx].clone()}
    return SimpleDataset(encodings)

def get_layer_masters(model, dataloader, layer_idx, top_k=10):
    model.eval()
    acts = []
    def hook_fn(module, input, output):
        acts.append(output.mean(dim=1).detach().cpu().numpy())
    
    # В distilgpt2 c_fc имеет размерность [768, 3072]
    target_layer = model.transformer.h[layer_idx].mlp.c_fc
    handle = target_layer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for batch in dataloader:
            model(input_ids=batch['input_ids'].to(DEVICE), attention_mask=batch['attention_mask'].to(DEVICE))
            if len(acts) >= 10: break
    handle.remove()
    
    acts = np.concatenate(acts, axis=0)
    G = nx.Graph()
    num_nodes = acts.shape[1]
    G.add_nodes_from(range(num_nodes))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = np.nan_to_num(np.abs(np.corrcoef(acts.T)), nan=0.0)
    
    # Агрессивная оптимизация для скорости
    MAX_DEGREE = 15
    CORR_THRESHOLD = 0.4
    capped_edges = []
    for i in range(num_nodes):
        row_corr = corr[i, i+1:]
        if len(row_corr) == 0: continue
        top_idx = np.argsort(row_corr)[-MAX_DEGREE:]
        for idx in top_idx:
            j = i + 1 + idx
            if row_corr[idx] > CORR_THRESHOLD:
                capped_edges.append((i, j, row_corr[idx]))
                
    G.add_weighted_edges_from(capped_edges)
    
    if len(capped_edges) == 0: return list(range(top_k)) # Fallback
    
    orc = OllivierRicci(G, alpha=0.5)
    Gr = orc.compute_ricci_curvature()
    curv = {n: Gr.nodes[n].get('ricciCurvature', 0.0) for n in G.nodes}
    
    sorted_m = sorted(curv.items(), key=lambda x: x[1], reverse=True)
    return [n for n, c in sorted_m[:top_k]]

def evaluate_perplexity(model, dataloader):
    model.eval()
    total_loss, total_batches = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            out = model(input_ids=batch['input_ids'].to(DEVICE), labels=batch['labels'].to(DEVICE))
            total_loss += out.loss.item()
            total_batches += 1
    return np.exp(total_loss / total_batches)

def train_all_mlps(model, dataloader, epochs, lr=5e-4):
    # Размораживаем ВСЕ c_fc и c_proj во всех слоях для выращивания
    for param in model.parameters(): param.requires_grad = False
    for i in range(len(model.transformer.h)):
        model.transformer.h[i].mlp.c_fc.weight.requires_grad = True
        model.transformer.h[i].mlp.c_proj.weight.requires_grad = True
        model.transformer.h[i].mlp.c_fc.bias.requires_grad = True

    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    for _ in range(epochs):
        for batch in dataloader:
            opt.zero_grad()
            out = model(input_ids=batch['input_ids'].to(DEVICE), labels=batch['labels'].to(DEVICE))
            out.loss.backward()
            opt.step()
    return model

if __name__ == "__main__":
    print("="*70)
    print("NEURAL SEED: РАСПРЕДЕЛЕННОЕ СЕМЯ (Все слои)")
    print("="*70)
    
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token
    dataset = get_json_dataset(tokenizer, num_samples=500)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    
    print("\n[1] Обучение Эталона (размораживаем все MLP слои, 2 эпохи)...")
    set_seed(42)
    oracle = GPT2LMHeadModel.from_pretrained('distilgpt2').to(DEVICE)
    oracle = train_all_mlps(oracle, train_loader, epochs=EPOCHS_ORACLE)
    ppl_oracle = evaluate_perplexity(oracle, train_loader)
    print(f"✅ Перплексия Эталона: {ppl_oracle:.2f}")
    
    print("\n[2] Извлечение распределенных мастеров...")
    num_layers = len(oracle.transformer.h)
    distributed_seed = {}
    total_seed_bytes = 0
    
    for i in range(num_layers):
        print(f"  Анализ слоя {i}/{num_layers-1}...")
        masters = get_layer_masters(oracle, train_loader, layer_idx=i, top_k=MASTERS_PER_LAYER)
        
        fc1_w = oracle.transformer.h[i].mlp.c_fc.weight.data[:, masters].clone().cpu()
        fc2_w = oracle.transformer.h[i].mlp.c_proj.weight.data[masters, :].clone().cpu()
        fc1_b = oracle.transformer.h[i].mlp.c_fc.bias.data[masters].clone().cpu()
        
        distributed_seed[i] = {"masters": masters, "fc1_w": fc1_w, "fc2_w": fc2_w, "fc1_b": fc1_b}
        total_seed_bytes += fc1_w.numel()*4 + fc2_w.numel()*4 + fc1_b.numel()*4
        
    print(f"  🌱 Общий размер распределенного семени: {total_seed_bytes/1024:.1f} KB")
    print(f"  📦 Оригинал: {sum(p.numel() for p in oracle.parameters())*4/1024:.1f} KB")
    print(f"  📉 Сжатие: {(sum(p.numel() for p in oracle.parameters())*4) / total_seed_bytes:.1f}x")

    print("\n[3] Выращивание из распределенного семени (1 эпоха)...")
    set_seed(42)
    grown = GPT2LMHeadModel.from_pretrained('distilgpt2').to(DEVICE)
    
    # Внедрение семени во все слои
    for i in range(num_layers):
        m = distributed_seed[i]["masters"]
        grown.transformer.h[i].mlp.c_fc.weight.data[:, m] = distributed_seed[i]["fc1_w"].to(DEVICE)
        grown.transformer.h[i].mlp.c_proj.weight.data[m, :] = distributed_seed[i]["fc2_w"].to(DEVICE)
        grown.transformer.h[i].mlp.c_fc.bias.data[m] = distributed_seed[i]["fc1_b"].to(DEVICE)
        
    grown = train_all_mlps(grown, train_loader, epochs=EPOCHS_GROW)
    ppl_grown = evaluate_perplexity(grown, train_loader)
    
    print(f"\n✅ Перплексия Распределенного Семени (1 эпоха): {ppl_grown:.2f}")
    print(f"   Отставание от Эталона (2 эпохи): {ppl_grown - ppl_oracle:+.2f}")
    
    print("\n" + "="*70)
    print("ВЫВОД: Если отставание <= 0.05, мы доказали, что восстановление")
    print("топологии на всех уровнях абстракции компенсирует недостаток эпох.")
    print("="*70)

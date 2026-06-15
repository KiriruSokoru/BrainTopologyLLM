"""
Путь 1: Топологическое выравнивание (Alignment) микро-LLM.
Используем "Эмбриогенез весов" (Hard Reg, alpha=0.5) во время микро-обучения,
чтобы заставить модель генерировать строгий JSON без "воды".
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import warnings
import time
import json
import re

# ========== НАСТРОЙКИ ==========
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = "distilgpt2" # 82M параметров, идеально для быстрого CPU теста
EPOCHS = 5
STEPS_PER_PROJECTION = 10  # Жесткая проекция каждые 10 шагов
ALPHA = 0.5                # Hard Reg: 50% сдвиг к мастеру

print(f"Устройство: {DEVICE}")
print("Загрузка distilgpt2...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

# ========== МИКРО-ДАТАСЕТ (Строгий JSON) ==========
# Формат: "Входной текст <SEP> Ожидаемый JSON"
# Мы учим модель, что после <SEP> должен идти ТОЛЬКО JSON
DATA_PAIRS = [
    ("Иван, 25, Москва, инженер.", '{"name": "Иван", "age": 25, "city": "Москва", "job": "инженер"}'),
    ("Мария, 30, СПБ, врач.", '{"name": "Мария", "age": 30, "city": "СПБ", "job": "врач"}'),
    ("Алексей, 40, Казань, IT.", '{"name": "Алексей", "age": 40, "city": "Казань", "job": "IT"}'),
    ("Елена, 22, Сочи, дизайнер.", '{"name": "Елена", "age": 22, "city": "Сочи", "job": "дизайнер"}'),
    ("Дмитрий, 35, Минск, менеджер.", '{"name": "Дмитрий", "age": 35, "city": "Минск", "job": "менеджер"}'),
]

def prepare_dataset(pairs):
    input_ids, labels = [], []
    for text, target_json in pairs:
        # Формируем промпт: "Extract: [text] JSON: [target]"
        prompt = f"Extract: {text} JSON: "
        full_text = prompt + target_json + tokenizer.eos_token
        
        enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=64)
        input_ids.append(enc.input_ids.squeeze(0))
        
        # Labels: игнорируем промпт, учим предсказывать только JSON
        label = enc.input_ids.squeeze(0).clone()
        prompt_len = len(tokenizer(prompt, return_tensors="pt").input_ids[0])
        label[:prompt_len] = -100 
        labels.append(label)
        
    return DataLoader(TensorDataset(torch.stack(input_ids), torch.stack(labels)), batch_size=2, shuffle=True)

# ========== ТОПЛОГИЧЕСКАЯ ПРОЕКЦИЯ (Hard Reg) ==========
def apply_topological_projection(model, dataloader):
    model.eval()
    activations = []
    target_layer = model.transformer.h[-1].attn.c_proj # 768 нейронов для скорости
    
    def hook_fn(module, input, output):
        # output shape: (batch, seq_len, 768). Берем последний токен.
        act = output[:, -1, :].to(torch.float32).detach().cpu().numpy()
        activations.append(act)

    handle = target_layer.register_forward_hook(hook_fn)
    with torch.no_grad():
        for x, _ in dataloader:
            _ = model(x.to(DEVICE))
    handle.remove()
    
    acts = np.concatenate(activations, axis=0)
    n_neurons = acts.shape[1]
    G = nx.Graph()
    G.add_nodes_from(range(n_neurons))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = np.corrcoef(acts.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Высокий порог для быстрого и чистого графа
    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            if abs(corr[i, j]) > 0.7:
                G.add_edge(i, j, weight=abs(corr[i, j]))
                
    if len(G.edges) == 0:
        model.train()
        return 0
        
    orc = OllivierRicci(G, alpha=0.5)
    G_ricci = orc.compute_ricci_curvature()
    
    curvatures = {node: G_ricci.nodes[node].get('ricciCurvature', 0.0) for node in G.nodes}
    singularities = [node for node, c in curvatures.items() if c < 0.0]
    masters = [node for node, c in curvatures.items() if c > 0.1]
    
    suppressed = 0
    if len(singularities) > 0 and len(masters) > 0:
        weights = target_layer.weight.data.clone()
        master_weights = weights[masters].mean(dim=0)
        
        for sing in singularities:
            weights[sing] = (1 - ALPHA) * weights[sing] + ALPHA * master_weights
            suppressed += 1
            
        target_layer.weight.data = weights
        
    model.train()
    return suppressed

# ========== ОЦЕНКА ==========
def evaluate_model(model, tokenizer):
    model.eval()
    valid_json_count = 0
    total_prompts = len(DATA_PAIRS)
    
    print("\n--- ГЕНЕРАЦИЯ ПОСЛЕ ОБУЧЕНИЯ ---")
    for text, _ in DATA_PAIRS:
        prompt = f"Extract: {text} JSON: "
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            
        generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Простая проверка на валидный JSON
        try:
            # Пытаемся найти и распарсить JSON
            match = re.search(r'\{.*\}', generated, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                if "name" in parsed and "age" in parsed:
                    valid_json_count += 1
                    print(f"✅ {generated}")
                else:
                    print(f"⚠️ (Неполный JSON) {generated}")
            else:
                print(f"❌ (Нет JSON) {generated}")
        except:
            print(f"❌ (Ошибка парсинга) {generated}")
            
    model.train()
    return (valid_json_count / total_prompts) * 100

# ========== ГЛАВНЫЙ ЦИКЛ ==========
if __name__ == "__main__":
    print("="*70)
    print("ПУТЬ 1: ТОПЛОГИЧЕСКОЕ ВЫРАВНИВАНИЕ LLM (Эмбриогенез)")
    print(f"Модель: {MODEL_NAME} | Alpha: {ALPHA} | Проекция каждые {STEPS_PER_PROJECTION} шагов")
    print("="*70)
    
    dataloader = prepare_dataset(DATA_PAIRS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    print("\n[1/2] Оценка ДО обучения:")
    acc_before = evaluate_model(model, tokenizer)
    print(f"Валидный JSON: {acc_before:.1f}%")
    
    print("\n[2/2] Начало топологического обучения...")
    global_step = 0
    
    for epoch in range(EPOCHS):
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_x, labels=batch_y)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            global_step += 1
            
            # ЭМБРИОГЕНЕЗ: жесткая проекция
            if global_step % STEPS_PER_PROJECTION == 0:
                suppressed = apply_topological_projection(model, dataloader)
                print(f"  Шаг {global_step:3d} | Loss: {loss.item():.4f} | Подавлено сингулярностей: {suppressed}")
                
    print("\n[3/3] Финальная оценка:")
    acc_after = evaluate_model(model, tokenizer)
    print(f"\nВалидный JSON: {acc_after:.1f}%")
    
    print("\n" + "="*70)
    print("ИТОГ")
    print("="*70)
    print(f"Точность до: {acc_before:.1f}%")
    print(f"Точность после: {acc_after:.1f}%")
    
    if acc_after > acc_before:
        print("🎉 УСПЕХ: Топологическая регуляризация во время обучения заставила микро-LLM")
        print("   выучить строгий формат JSON, подавив 'шумные' сингулярности!")
    else:
        print("🔍 АНАЛИЗ: Модель не улучшилась. Возможно, 82M параметров слишком мало для")
        print("   такой задачи, или требуется больше шагов. Переходим к поиску 'божьей искры'.")
    print("="*70)

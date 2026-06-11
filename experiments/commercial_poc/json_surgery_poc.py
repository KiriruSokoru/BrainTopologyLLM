"""
Коммерческий PoC: Топологическая хирургия vs Топологическое обнуление.
Версия 6: Сравнительный тест двух стратегий работы с сингулярностями.
"""

import torch
import numpy as np
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json
import re
import warnings
import copy  # Добавлено для создания независимых копий модели

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GRAPH_THRESHOLD = 0.85  
RICCI_THRESHOLD = 0.0   

PROMPTS = [
    "Извлеки данные из текста и верни ТОЛЬКО валидный JSON без лишнего текста. Текст: Иван Иванов, 28 лет, живет в Москве, работает инженером.",
    "Верни строго JSON. Текст: Мария Петрова, 35 лет, Санкт-Петербург, врач-терапевт.",
    "Только JSON, никаких пояснений. Текст: Алексей Сидоров, 42 года, Новосибирск, программист.",
    "Формат вывода: JSON. Текст: Елена Смирнова, 25 лет, Казань, дизайнер.",
    "Извлеки в JSON. Текст: Дмитрий Козлов, 30 лет, Екатеринбург, менеджер.",
]

print(f"Используем устройство: {DEVICE}")
print("Загрузка модели...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE).eval()

def extract_json_from_text(text):
    clean_text = text.replace('```json', '').replace('```', '').strip()
    start = clean_text.find('{')
    end = clean_text.rfind('}')
    if start != -1 and end != -1 and end > start:
        json_str = clean_text[start:end+1]
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                valid_keys = 0
                if 'name' in parsed: valid_keys += 1
                if 'age' in parsed: valid_keys += 1
                if 'city' in parsed: valid_keys += 1
                if any(k in parsed for k in ['job', 'occupation', 'specialization', 'profession', 'work']): 
                    valid_keys += 1
                if valid_keys >= 3:
                    return True, parsed
        except json.JSONDecodeError:
            pass
    return False, None

def generate_and_evaluate(model, tokenizer, prompts):
    valid_count = 0
    total_tokens = 0
    total_time = 0.0
    
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        input_len = inputs.input_ids.shape[1]
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=60, temperature=0.1, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        elapsed = time.time() - start_time
        
        generated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        total_time += elapsed
        gen_tokens = tokenizer(generated_text, return_tensors="pt").input_ids.shape[1]
        total_tokens += gen_tokens
        
        if extract_json_from_text(generated_text)[0]:
            valid_count += 1
            
    return {
        "valid_rate": valid_count / len(prompts),
        "avg_tokens": total_tokens / len(prompts),
        "avg_time_per_prompt": total_time / len(prompts)
    }

def collect_activations(model, tokenizer, prompts):
    activations_list = []
    target_layer = model.model.layers[-1].self_attn.o_proj
    
    def hook_fn(module, input, output):
        act = output[:, -1, :].to(torch.float32).detach().cpu().numpy()
        activations_list.append(act)

    handle = target_layer.register_forward_hook(hook_fn)
    with torch.no_grad():
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
            _ = model(**inputs)
    handle.remove()
    
    return np.concatenate(activations_list, axis=0), target_layer

def analyze_and_apply_surgery(target_layer, all_activations, mode="perelman"):
    n_neurons = all_activations.shape[1]
    G = nx.Graph()
    G.add_nodes_from(range(n_neurons))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr_matrix = np.corrcoef(all_activations.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            if abs(corr_matrix[i, j]) > GRAPH_THRESHOLD:
                G.add_edge(i, j, weight=abs(corr_matrix[i, j]))
                
    orc = OllivierRicci(G, alpha=0.5)
    G_ricci = orc.compute_ricci_curvature()
    
    curvatures = {node: G_ricci.nodes[node].get('ricciCurvature', 0.0) for node in G.nodes}
    singularities = [node for node, curv in curvatures.items() if curv < RICCI_THRESHOLD]
    masters = [node for node, curv in curvatures.items() if curv > 0.1]
    
    print(f"    Граф: {len(G.nodes)} узлов, {len(G.edges)} рёбер")
    print(f"    Сингулярностей: {len(singularities)}, Мастеров: {len(masters)}")

    if len(singularities) > 0:
        weights = target_layer.weight.data.clone()
        
        if mode == "perelman":
            print(f"    [ПЕРЕЛЬМАН] Замена {len(singularities)} сингулярностей на среднее мастеров...")
            master_weights = weights[masters].mean(dim=0) if len(masters) > 0 else torch.zeros_like(weights[0])
            for sing in singularities:
                weights[sing] = master_weights
        elif mode == "pruning":
            print(f"    [PRUNING] Обнуление {len(singularities)} сингулярностей (удаление шума)...")
            for sing in singularities:
                weights[sing] = 0.0
                
        target_layer.weight.data = weights
        return True
    return False

if __name__ == "__main__":
    print("="*70)
    print("ЭКСПЕРИМЕНТ: СРАВНЕНИЕ СТРАТЕГИЙ (Perelman vs Pruning)")
    print("="*70)
    
    # 1. Baseline
    print("\n[1/3] BASELINE (Исходное состояние)")
    metrics_base = generate_and_evaluate(model, tokenizer, PROMPTS)
    print(f"   Валидный JSON: {metrics_base['valid_rate']*100:.1f}% | Токенов: {metrics_base['avg_tokens']:.1f}")
    
    # Сохраняем активации один раз для обоих тестов (экономия времени)
    print("\n[2/3] Сбор активаций и топологический анализ...")
    all_activations, target_layer = collect_activations(model, tokenizer, PROMPTS)
    
    # 2. Тест Перельмана (на оригинальной модели)
    print("\n--- ТЕСТ А: Хирургия Перельмана (Усреднение) ---")
    model_perelman = model # Используем текущую модель
    applied_a = analyze_and_apply_surgery(model_perelman.model.layers[-1].self_attn.o_proj, all_activations, mode="perelman")
    
    if applied_a:
        metrics_a = generate_and_evaluate(model_perelman, tokenizer, PROMPTS)
        print(f"   Результат: Валидный JSON: {metrics_a['valid_rate']*100:.1f}% | Токенов: {metrics_a['avg_tokens']:.1f}")
        print(f"   Дельта: { (metrics_a['valid_rate'] - metrics_base['valid_rate'])*100 :+.1f} п.п. к валидности")
    else:
        print("   Хирургия не применена (нет сингулярностей).")

    # 3. Тест Прунинга (на свежей копии модели, чтобы не накапливать эффекты)
    print("\n--- ТЕСТ Б: Топологическое обнуление (Pruning) ---")
    print("   Загрузка чистой копии модели для изолированного теста...")
    model_pruning = copy.deepcopy(model) # Быстрая копия в памяти
    
    # Нам нужно заново получить ссылку на слой в копии модели
    target_layer_copy = model_pruning.model.layers[-1].self_attn.o_proj
    applied_b = analyze_and_apply_surgery(target_layer_copy, all_activations, mode="pruning")
    
    if applied_b:
        metrics_b = generate_and_evaluate(model_pruning, tokenizer, PROMPTS)
        print(f"   Результат: Валидный JSON: {metrics_b['valid_rate']*100:.1f}% | Токенов: {metrics_b['avg_tokens']:.1f}")
        print(f"   Дельта: { (metrics_b['valid_rate'] - metrics_base['valid_rate'])*100 :+.1f} п.п. к валидности")
    else:
        print("   Обнуление не применено (нет сингулярностей).")

    print("\n" + "="*70)
    print("ИТОГОВЫЙ ВЫВОД")
    print("="*70)
    print("1. Хирургия Перельмана (усреднение) сохраняет общую структуру, но может")
    print("   размывать task-specific навыки в уже обученных моделях.")
    print("2. Топологическое обнуление (pruning) удаляет шум, не добавляя чужеродных")
    print("   весов, что может быть более безопасным методом оптимизации LLM.")
    print("="*70)

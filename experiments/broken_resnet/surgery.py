import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from ricci_metrics import RicciCurvature

class RicciSurgery:
    """Хирургия Перельмана: замена сингулярностей на стандартные колпачки"""
    
    def __init__(self, model, device='cuda', threshold_sing=-0.25, threshold_master=0.15):
        self.model = model
        self.device = device
        self.threshold_sing = threshold_sing
        self.threshold_master = threshold_master
        self.surgery_log = []
    
    def perform_surgery(self, dataloader, verbose=True):
        """Проводит операцию: диагностика → замена сингулярностей"""
        
        # 1. Диагностика
        print("  🔍 Проводим предоперационную диагностику...")
        ricci = RicciCurvature(self.model, self.device)
        stats = ricci.diagnose(dataloader, num_batches=5)
        
        if verbose:
            print(f"\n  📊 ДИАГНОСТИКА ПЕРЕД ОПЕРАЦИЕЙ:")
            print(f"     Мастеров: {stats['masters']:.0f}")
            print(f"     Стажёров: {stats['juniors']:.0f}")
            print(f"     Сингулярностей: {stats['singularities']:.0f}")
        
        # 2. Для каждого слоя с сингулярностями - операция
        total_replaced = 0
        total_masters_used = 0
        
        layers_with_sing = [name for name, s in stats['by_layer'].items() if s['singularities'] > 0]
        
        if not layers_with_sing:
            print("  ✅ Сингулярностей не найдено, операция не требуется")
            return stats, []
        
        pbar = tqdm(layers_with_sing, desc="🔪 Хирургия по слоям", leave=False, unit='layer')
        
        for layer_name in pbar:
            layer_stats = stats['by_layer'][layer_name]
            pbar.set_postfix({
                'слой': layer_name.split('.')[-1],
                'синг': f'{layer_stats["singularities"]:.0f}'
            })
            
            # Получаем модуль
            module = self._get_module_by_name(layer_name)
            if module is None or not isinstance(module, nn.Conv2d):
                continue
            
            # Получаем кривизну для каждого канала
            per_channel_ricci = ricci.compute_per_channel_ricci(dataloader, layer_name, num_batches=3)
            if per_channel_ricci is None:
                continue
            
            singular_idx = np.where(per_channel_ricci < self.threshold_sing)[0]
            master_idx = np.where(per_channel_ricci > self.threshold_master)[0]
            
            if len(master_idx) == 0:
                master_weights = module.weight.data.mean(dim=0)
                master_idx = list(range(module.weight.shape[0]))
            else:
                master_weights = module.weight.data[master_idx].mean(dim=0)
            
            # Заменяем сингулярные каналы
            for idx in singular_idx:
                noise = torch.randn_like(master_weights) * 0.01
                module.weight.data[idx] = master_weights + noise
                total_replaced += 1
            
            total_masters_used += len(master_idx)
            
            self.surgery_log.append({
                'layer': layer_name,
                'singularities_replaced': int(len(singular_idx)),
                'masters_used': int(len(master_idx)),
                'singularity_indices': singular_idx.tolist(),
                'master_indices': master_idx.tolist()
            })
        
        if verbose:
            print(f"\n  ✅ ХИРУРГИЯ ЗАВЕРШЕНА")
            print(f"     Заменено сингулярностей: {total_replaced}")
            print(f"     Использовано мастеров: {total_masters_used}")
        
        return stats, self.surgery_log
    
    def _get_module_by_name(self, name):
        """Получает модуль по строковому имени"""
        parts = name.split('.')
        module = self.model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part, None)
            if module is None:
                return None
        return module

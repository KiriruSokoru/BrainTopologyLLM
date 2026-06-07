import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from tqdm import tqdm

class RicciCurvature:
    """
    Вычисляет кривизну Риччи для свёрточных слоёв
    Только forward pass — без backward хуков
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
    
    def diagnose(self, dataloader, num_batches=5):
        """Диагностирует сеть через анализ активаций"""
        self.model.eval()
        
        all_stats = {
            'masters': 0,
            'juniors': 0,
            'singularities': 0,
            'by_layer': defaultdict(lambda: {'masters': 0, 'juniors': 0, 'singularities': 0, 'ricci_mean': []})
        }
        
        activations = {}
        hooks = []
        
        def forward_hook(name):
            def hook(module, input, output):
                activations[name] = output.clone().detach()
            return hook
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(forward_hook(name)))
        
        pbar = tqdm(range(num_batches), desc="Сбор активаций", leave=False, unit='batch')
        
        for batch_idx in pbar:
            try:
                data, target = next(iter(dataloader))
            except StopIteration:
                break
            
            data = data.to(self.device)
            activations.clear()
            
            with torch.no_grad():
                _ = self.model(data)
            
            layer_count = 0
            for name, act in activations.items():
                batch, channels, h, w = act.shape
                
                # Усредняем по пространству
                act_flat = act.view(batch, channels, -1).mean(dim=-1)
                
                # Нормализуем
                act_norm = (act_flat - act_flat.mean(dim=0)) / (act_flat.std(dim=0) + 1e-8)
                act_np = act_norm.cpu().numpy()
                
                ric = np.zeros(channels)
                for c in range(channels):
                    neighbors = [n for n in range(max(0, c-3), min(channels, c+4)) if n != c]
                    if not neighbors:
                        ric[c] = 0
                        continue
                    
                    corr_sum = 0
                    for n in neighbors:
                        corr = np.corrcoef(act_np[:, c], act_np[:, n])[0, 1]
                        if not np.isnan(corr):
                            corr_sum += corr
                    ric[c] = corr_sum / len(neighbors)
                
                masters = np.sum(ric > 0.15)
                singularities = np.sum(ric < -0.25)
                juniors = len(ric) - masters - singularities
                
                all_stats['by_layer'][name]['masters'] += masters
                all_stats['by_layer'][name]['juniors'] += juniors
                all_stats['by_layer'][name]['singularities'] += singularities
                all_stats['by_layer'][name]['ricci_mean'].append(ric.mean())
                layer_count += 1
            
            pbar.set_postfix({'слоёв': layer_count})
        
        # Усредняем
        n = num_batches
        for layer in all_stats['by_layer']:
            stats = all_stats['by_layer'][layer]
            stats['masters'] /= n
            stats['juniors'] /= n
            stats['singularities'] /= n
            stats['ricci_mean'] = np.mean(stats['ricci_mean'])
            
            all_stats['masters'] += stats['masters']
            all_stats['juniors'] += stats['juniors']
            all_stats['singularities'] += stats['singularities']
        
        for hook in hooks:
            hook.remove()
        
        return all_stats
    
    def compute_per_channel_ricci(self, dataloader, layer_name, num_batches=3):
        """Вычисляет кривизну для каждого канала слоя"""
        self.model.eval()
        
        activations = []
        hooks = []
        
        def forward_hook(name):
            def hook(module, input, output):
                if name == layer_name:
                    activations.append(output.clone().detach())
            return hook
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(forward_hook(name)))
        
        for _ in range(num_batches):
            try:
                data, _ = next(iter(dataloader))
            except StopIteration:
                break
            data = data.to(self.device)
            with torch.no_grad():
                _ = self.model(data)
        
        for hook in hooks:
            hook.remove()
        
        if not activations:
            return None
        
        all_acts = torch.cat(activations, dim=0)
        batch, channels, h, w = all_acts.shape
        
        act_flat = all_acts.view(batch, channels, -1).mean(dim=-1)
        act_norm = (act_flat - act_flat.mean(dim=0)) / (act_flat.std(dim=0) + 1e-8)
        act_np = act_norm.cpu().numpy()
        
        ric = np.zeros(channels)
        for c in range(channels):
            neighbors = [n for n in range(max(0, c-3), min(channels, c+4)) if n != c]
            if not neighbors:
                ric[c] = 0
                continue
            
            corr_sum = 0
            for n in neighbors:
                corr = np.corrcoef(act_np[:, c], act_np[:, n])[0, 1]
                if not np.isnan(corr):
                    corr_sum += corr
            ric[c] = corr_sum / len(neighbors)
        
        return ric

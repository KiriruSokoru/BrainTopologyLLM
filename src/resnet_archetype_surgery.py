#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Хирургия Перельмана для ResNet через архетип мастера.
Версия 4.0 — с правильной слоевой группировкой и чекпоинтами.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
import argparse
import warnings

# ===== АВТОМАТИЧЕСКИЙ КОНТРОЛЬ НАГРУЗКИ =====
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from load_controller import get_controller, patch_dataloaders
    controller = get_controller()
    patch_dataloaders()
    print("🎛️ LoadController активирован")
except ImportError:
    print("⚠️ LoadController не найден, работаем без авто-контроля")
    torch.set_num_threads(4)
    os.environ['OMP_NUM_THREADS'] = '4'

from checkpoint_manager import CheckpointManager

# Парсинг аргументов
parser = argparse.ArgumentParser()
parser.add_argument('--force_restart', action='store_true', 
                    help='Принудительно начать эксперимент заново')
parser.add_argument('--skip_diagnosis', action='store_true',
                    help='Пропустить диагностику (для отладки)')
parser.add_argument('--skip_surgery', action='store_true',
                    help='Пропустить хирургию')
args = parser.parse_args()

# Настройки
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Работаем на {DEVICE}")


# ===============================
# 1. Базовый блок с измерением кривизны
# ===============================

class BasicBlockWithRicci(nn.Module):
    """Блок ResNet с возможностью измерения бригадной кривизны"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out
    
    def measure_curvature_on_input(self, x):
        """Измеряем кривизну бригады на заданном входе x"""
        x = x[:32].to(DEVICE)
        x.requires_grad_(True)
        
        identity = self.shortcut(x)
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out = out2 + identity
        
        grad_conv = torch.autograd.grad(out.sum(), out2, retain_graph=True)[0]
        grad_skip = torch.autograd.grad(out.sum(), identity, retain_graph=True)[0]
        
        importance_conv = grad_conv.abs().mean().item()
        importance_skip = grad_skip.abs().mean().item()
        
        eps = 1e-8
        ricci = importance_skip / (importance_conv + importance_skip + eps)
        
        return 2 * ricci - 1


# ===============================
# 2. ResNet с бригадами
# ===============================

class ResNetWithBrigades(nn.Module):
    """ResNet-18, где каждый BasicBlock — отдельная бригада"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Собираем все бригады в список
        self.brigades = []
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                self.brigades.append(block)
        
        # Группировка бригад по слоям (для диагностики)
        self.layer_brigades = {
            0: [0, 1],   # layer1 (64 канала)
            1: [2, 3],   # layer2 (128 каналов)
            2: [4, 5],   # layer3 (256 каналов)
            3: [6, 7]    # layer4 (512 каналов)
        }
        
        print(f"🏗️ Создано {len(self.brigades)} бригад")
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlockWithRicci(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlockWithRicci(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def diagnose_curvature(self, dataloader, num_batches=30):
        """Диагностика всех бригад"""
        self.eval()
        brigade_curvatures = [[] for _ in range(len(self.brigades))]
        
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break
                
                x = x.to(DEVICE)
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                
                brigade_idx = 0
                
                for block in self.layer1:
                    block_input = x.clone().detach()
                    x = block(block_input)
                    with torch.enable_grad():
                        ric = block.measure_curvature_on_input(block_input)
                    brigade_curvatures[brigade_idx].append(ric)
                    brigade_idx += 1
                
                for block in self.layer2:
                    block_input = x.clone().detach()
                    x = block(block_input)
                    with torch.enable_grad():
                        ric = block.measure_curvature_on_input(block_input)
                    brigade_curvatures[brigade_idx].append(ric)
                    brigade_idx += 1
                
                for block in self.layer3:
                    block_input = x.clone().detach()
                    x = block(block_input)
                    with torch.enable_grad():
                        ric = block.measure_curvature_on_input(block_input)
                    brigade_curvatures[brigade_idx].append(ric)
                    brigade_idx += 1
                
                for block in self.layer4:
                    block_input = x.clone().detach()
                    x = block(block_input)
                    with torch.enable_grad():
                        ric = block.measure_curvature_on_input(block_input)
                    brigade_curvatures[brigade_idx].append(ric)
                    brigade_idx += 1
        
        results = []
        for idx, curv_list in enumerate(brigade_curvatures):
            mean_ric = np.mean(curv_list) if curv_list else 0.0
            std_ric = np.std(curv_list) if curv_list else 0.0
            
            # Определяем слой бригады
            layer = None
            for l, brigades in self.layer_brigades.items():
                if idx in brigades:
                    layer = l + 1
                    break
            
            if mean_ric > 0.7:
                status = "🔥 ХАОТИК"
            elif mean_ric < -0.7:
                status = "⭐ МАСТЕР"
            else:
                status = "📚 СТАЖЁР"
            
            results.append({
                'brigade': idx,
                'layer': layer,
                'curvature': mean_ric,
                'std': std_ric,
                'status': status
            })
        
        return results


# ===============================
# 3. Хирург с правильной слоевой группировкой
# ===============================

class ArchetypalSurgeon:
    """Хирург, заменяющий хаотиков на архетип мастера (внутри каждого слоя)"""
    
    def __init__(self, model):
        self.model = model
        self.archetype_by_layer = {}  # {layer_idx: archetype_weights}
        self.master_indices = []
        self.singularity_indices = []
        
        # Группировка бригад по слоям
        self.layer_brigades = {
            0: [0, 1],   # layer1 (64 канала)
            1: [2, 3],   # layer2 (128 каналов)
            2: [4, 5],   # layer3 (256 каналов)
            3: [6, 7]    # layer4 (512 каналов)
        }
    
    def build_archetype(self, diagnosis):
        """Строит архетип для каждого слоя из его мастеров"""
        print(f"\n🏆 Построение архетипов по слоям...")
        
        self.archetype_by_layer = {}
        self.master_indices = []
        
        for layer_idx, brigade_ids in self.layer_brigades.items():
            # Бригады этого слоя
            layer_brigades = [b for b in diagnosis if b['brigade'] in brigade_ids]
            
            if not layer_brigades:
                continue
            
            # Сортируем по кривизне (от меньшей к большей)
            sorted_layer = sorted(layer_brigades, key=lambda x: x['curvature'])
            
            # Берём лучшую бригаду в этом слое как мастера
            num_masters = max(1, len(sorted_layer) // 2)
            masters_in_layer = sorted_layer[:num_masters]
            
            for master in masters_in_layer:
                self.master_indices.append(master['brigade'])
            
            # Строим архетип для этого слоя
            if masters_in_layer:
                archetype_weights = {
                    'conv1.weight': [], 'conv2.weight': [],
                    'bn1.weight': [], 'bn2.weight': []
                }
                
                for master in masters_in_layer:
                    brigade = self.model.brigades[master['brigade']]
                    archetype_weights['conv1.weight'].append(brigade.conv1.weight.data)
                    archetype_weights['conv2.weight'].append(brigade.conv2.weight.data)
                    archetype_weights['bn1.weight'].append(brigade.bn1.weight.data)
                    archetype_weights['bn2.weight'].append(brigade.bn2.weight.data)
                
                # Усредняем
                layer_archetype = {}
                for key, weights_list in archetype_weights.items():
                    layer_archetype[key] = torch.stack(weights_list).mean(dim=0)
                
                self.archetype_by_layer[layer_idx] = layer_archetype
                
                print(f"   Слой {layer_idx+1} (бригады {brigade_ids}): "
                      f"{len(masters_in_layer)} мастер(ов) → архетип построен")
        
        return self.archetype_by_layer
    
    def apply_surgery(self, diagnosis, curvature_threshold=0.7):
        """Заменяет сингулярности на архетип того же слоя"""
        self.singularity_indices = []
        
        print(f"\n🔪 Поиск сингулярностей...")
        
        for layer_idx, brigade_ids in self.layer_brigades.items():
            # Находим сингулярности в этом слое
            singularities_in_layer = [
                b for b in diagnosis 
                if b['brigade'] in brigade_ids and b['curvature'] > curvature_threshold
            ]
            
            if not singularities_in_layer:
                continue
            
            self.singularity_indices.extend([b['brigade'] for b in singularities_in_layer])
            
            if layer_idx not in self.archetype_by_layer:
                print(f"   ⚠️ Слой {layer_idx+1}: нет архетипа, пропускаем")
                continue
            
            print(f"\n   Слой {layer_idx+1}: {len(singularities_in_layer)} сингулярностей")
            layer_archetype = self.archetype_by_layer[layer_idx]
            
            for b in singularities_in_layer:
                b_idx = b['brigade']
                brigade = self.model.brigades[b_idx]
                
                for param_name, archetype_value in layer_archetype.items():
                    if 'conv1' in param_name:
                        target = brigade.conv1.weight
                    elif 'conv2' in param_name:
                        target = brigade.conv2.weight
                    elif 'bn1' in param_name:
                        target = brigade.bn1.weight
                    else:
                        target = brigade.bn2.weight
                    
                    # Архетип + малый шум
                    noise_scale = 0.01 * archetype_value.std()
                    noise = torch.randn_like(archetype_value) * noise_scale
                    target.data = archetype_value.clone() + noise
                    target.requires_grad = False
                
                # Замораживаем BN
                for param in brigade.bn1.parameters():
                    param.requires_grad = False
                for param in brigade.bn2.parameters():
                    param.requires_grad = False
                
                print(f"      ✂️ Бригада {b_idx} (кривизна {b['curvature']:.3f}) → заменена")
        
        if not self.singularity_indices:
            print("\n✅ Сингулярностей не найдено — хирургия не требуется")
        
        return self.singularity_indices
    
    def get_frozen_brigades(self):
        """Возвращает список замороженных бригад"""
        frozen = []
        for b_idx in self.singularity_indices:
            if b_idx < len(self.model.brigades):
                brigade = self.model.brigades[b_idx]
                if not brigade.conv1.weight.requires_grad:
                    frozen.append(b_idx)
        return frozen


# ===============================
# 4. Функции обучения
# ===============================

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x, y in tqdm(dataloader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, pred = out.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating", leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            
            total_loss += loss.item()
            _, pred = out.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
    
    return total_loss / len(dataloader), correct / total


def print_diagnosis(diagnosis):
    """Красиво печатает диагностику по слоям"""
    print("\n" + "="*70)
    print("🩺 ДИАГНОСТИКА БРИГАД ПО СЛОЯМ")
    print("="*70)
    
    current_layer = 1
    print(f"\n📁 СЛОЙ {current_layer} (каналы: 64):")
    
    for d in diagnosis:
        if d['layer'] != current_layer:
            current_layer = d['layer']
            channel_size = [64, 128, 256, 512][current_layer - 1]
            print(f"\n📁 СЛОЙ {current_layer} (каналы: {channel_size}):")
        
        icon = "⭐" if "МАСТЕР" in d['status'] else "🔥" if "ХАОТИК" in d['status'] else "📚"
        print(f"   {icon} Бригада {d['brigade']}: кривизна = {d['curvature']:.4f} ± {d['std']:.4f}")
    
    print("="*70)


# ===============================
# 5. Главная функция
# ===============================

def main():
    print("\n" + "="*60)
    print("🏥 ХИРУРГИЯ ПЕРЕЛЬМАНА ДЛЯ RESNET")
    print("="*60)
    
    # Загрузка данных
    print("\n📦 Загрузка CIFAR-10...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    import multiprocessing
    num_workers = min(4, multiprocessing.cpu_count() // 2)
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, 
        num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda')
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False,
        num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda')
    )
    
    print(f"📊 DataLoader: {num_workers} воркеров, batch_size=128")
    
    # Создание модели и чекпоинт-менеджера
    model = ResNetWithBrigades(num_classes=10).to(DEVICE)
    ckpt = CheckpointManager('resnet_cifar10_surgery')
    
    # Проверка на принудительный перезапуск
    if args.force_restart:
        print("\n⚠️ Принудительный перезапуск: удаляем старые чекпоинты")
        for stage in ['warmup_epoch10', 'diagnosis', 'surgery', 'relax_epoch25', 'relax_epoch50', 'final']:
            ckpt.delete_checkpoint(stage)
    
    # Восстановление или свежий старт
    latest_stage = ckpt.get_latest_stage()
    start_stage = 'fresh'
    
    if latest_stage and not args.force_restart:
        print(f"\n🔄 Найден сохранённый прогресс: {latest_stage}")
        resume = input("Продолжить с последнего чекпоинта? (y/n): ").lower()
        if resume == 'y':
            start_stage = latest_stage
    
    criterion = nn.CrossEntropyLoss()
    
    # ========== ЭТАП 0: РАЗОГРЕВ ==========
    if start_stage == 'fresh' or start_stage == 'warmup_epoch10':
        print("\n🔥 ЭТАП 0: РАЗОГРЕВ (10 эпох)")
        
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        
        for epoch in range(10):
            train_loss, train_acc = train_epoch(model, trainloader, optimizer, criterion, DEVICE)
            test_loss, test_acc = evaluate(model, testloader, criterion, DEVICE)
            scheduler.step()
            print(f"Epoch {epoch+1}: Train Acc = {train_acc:.3f}, Test Acc = {test_acc:.3f}")
        
        baseline_acc = test_acc
        print(f"\n📊 Baseline точность: {baseline_acc:.3f} ({baseline_acc*100:.1f}%)")
        
        ckpt.save('warmup_epoch10', model, optimizer, epoch=9, 
                  metrics={'baseline_acc': baseline_acc})
        start_stage = 'diagnosis'
    
    # ========== ЭТАП 1: ДИАГНОСТИКА ==========
    if start_stage == 'diagnosis' and not args.skip_diagnosis:
        print("\n🩺 ЭТАП 1: ДИАГНОСТИКА БРИГАД")
        
        # Загружаем модель
        if latest_stage in ['warmup_epoch10', 'diagnosis'] and not args.force_restart:
            checkpoint = ckpt.load('warmup_epoch10')
            model.load_state_dict(checkpoint['model_state'])
        
        diagnosis = model.diagnose_curvature(trainloader, num_batches=30)
        print_diagnosis(diagnosis)
        
        ckpt.save('diagnosis', model, extra={'diagnosis': diagnosis}, 
                  metrics={'num_brigades': len(diagnosis)})
        start_stage = 'surgery'
    
    # ========== ЭТАП 2: ХИРУРГИЯ ==========
    if start_stage == 'surgery' and not args.skip_surgery:
        print("\n🏥 ЭТАП 2: ХИРУРГИЯ ПЕРЕЛЬМАНА")
        
        # Загружаем диагностику
        if 'diagnosis' not in locals():
            checkpoint = ckpt.load('diagnosis')
            diagnosis = checkpoint['extra']['diagnosis']
        
        surgeon = ArchetypalSurgeon(model)
        surgeon.build_archetype(diagnosis)
        singularities = surgeon.apply_surgery(diagnosis, curvature_threshold=0.7)
        
        ckpt.save('surgery', model, 
                  extra={'singularities': singularities, 'masters': surgeon.master_indices},
                  metrics={'num_singularities': len(singularities)})
        start_stage = 'relax_epoch25'
    
    # ========== ЭТАП 3: РЕЛАКСАЦИЯ ==========
    if start_stage in ['relax_epoch25', 'relax_epoch50']:
        print("\n🧘 ЭТАП 3: РЕЛАКСАЦИЯ (50 эпох)")
        
        # Загружаем модель
        if 'singularities' not in locals():
            if ckpt.stage_exists('surgery'):
                checkpoint = ckpt.load('surgery')
                model.load_state_dict(checkpoint['model_state'])
                singularities = checkpoint['extra']['singularities']
            else:
                singularities = []
        
        # Если нет сингулярностей, релаксация не нужна
        if not singularities:
            print("\n✅ Сингулярностей нет — релаксация не требуется")
            start_stage = 'final'
        else:
            optimizer2 = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)
            scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=50)
            
            start_epoch = 0
            if start_stage == 'relax_epoch25' and ckpt.stage_exists('relax_epoch25'):
                ch = ckpt.load('relax_epoch25')
                model.load_state_dict(ch['model_state'])
                if ch['optimizer_state']:
                    optimizer2.load_state_dict(ch['optimizer_state'])
                start_epoch = ch.get('epoch', 0) + 1 if ch.get('epoch') else 0
                print(f"   Продолжаем с эпохи {start_epoch}")
            elif start_stage == 'relax_epoch50':
                final_check = ckpt.load('final')
                print(f"\n✅ Эксперимент уже завершён!")
                print(f"   Итоговая точность: {final_check['metrics']['final_acc']:.3f}")
                return
            
            for epoch in range(start_epoch, 50):
                train_loss, train_acc = train_epoch(model, trainloader, optimizer2, criterion, DEVICE)
                test_loss, test_acc = evaluate(model, testloader, criterion, DEVICE)
                scheduler2.step()
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}: Test Acc = {test_acc:.3f}")
                
                # Сохраняем на 25 эпохе
                if epoch == 24:
                    ckpt.save('relax_epoch25', model, optimizer2, epoch=epoch, 
                              metrics={'relax_acc': test_acc})
                
                # Размораживаем на 25 эпохе
                if epoch == 24:
                    print("\n🔓 Размораживаем прооперированные бригады...")
                    for b_idx in singularities:
                        brigade = model.brigades[b_idx]
                        for param in brigade.parameters():
                            param.requires_grad = True
            
            final_acc = test_acc
            
            # Загружаем baseline для сравнения
            warmup = ckpt.load('warmup_epoch10')
            baseline_acc = warmup['metrics']['baseline_acc']
            
            print(f"\n✅ Итоговая точность после хирургии: {final_acc:.3f} ({final_acc*100:.1f}%)")
            print(f"📉 Изменение относительно baseline ({baseline_acc:.3f}): {final_acc - baseline_acc:+.3f}")
            
            ckpt.save('relax_epoch50', model, optimizer2, epoch=49, 
                      metrics={'final_acc': final_acc, 'baseline_acc': baseline_acc})
            start_stage = 'final'
    
    # ========== ЭТАП 4: ФИНАЛ ==========
    if start_stage == 'final':
        print("\n💾 Сохраняем финальные результаты...")
        
        checkpoint = ckpt.load('relax_epoch50') if ckpt.stage_exists('relax_epoch50') else ckpt.load('warmup_epoch10')
        final_acc = checkpoint['metrics'].get('final_acc', checkpoint['metrics'].get('baseline_acc'))
        baseline_acc = checkpoint['metrics'].get('baseline_acc', final_acc)
        
        diagnosis_check = ckpt.load('diagnosis') if ckpt.stage_exists('diagnosis') else None
        diagnosis = diagnosis_check['extra']['diagnosis'] if diagnosis_check else []
        
        surgery_check = ckpt.load('surgery') if ckpt.stage_exists('surgery') else None
        singularities = surgery_check['extra']['singularities'] if surgery_check else []
        masters = surgery_check['extra']['masters'] if surgery_check else []
        
        results = {
            'baseline_acc': baseline_acc,
            'final_acc': final_acc,
            'singularities': singularities,
            'masters': masters,
            'diagnosis': diagnosis
        }
        
        os.makedirs('results', exist_ok=True)
        torch.save(results, 'results/resnet_archetype_surgery.pt')
        print("💾 Результаты сохранены в results/resnet_archetype_surgery.pt")
        
        # Визуализация
        if diagnosis:
            try:
                plt.figure(figsize=(12, 6))
                brigades = [d['brigade'] for d in diagnosis]
                curvatures = [d['curvature'] for d in diagnosis]
                colors = ['red' if c > 0.7 else 'green' if c < -0.7 else 'steelblue' for c in curvatures]
                
                bars = plt.bar(brigades, curvatures, color=colors, edgecolor='black')
                plt.axhline(y=0.7, color='red', linestyle='--', linewidth=2, label='Порог хаотика (0.7)')
                plt.axhline(y=-0.7, color='green', linestyle='--', linewidth=2, label='Порог мастера (-0.7)')
                plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
                
                # Подписи слоёв
                for i, d in enumerate(diagnosis):
                    if d['layer'] and d['layer'] > 1 and diagnosis[i-1]['layer'] != d['layer']:
                        plt.axvline(x=i-0.5, color='gray', linestyle=':', alpha=0.5)
                
                plt.xlabel('Номер бригады', fontsize=12)
                plt.ylabel('Кривизна Риччи', fontsize=12)
                plt.title('Диагностика бригад ResNet-18 на CIFAR-10\n' +
                          'Хаотики (>0.7) | Мастера (<-0.7) | Стажёры', fontsize=14)
                plt.legend(loc='upper right')
                plt.tight_layout()
                plt.savefig('results/brigade_curvature.png', dpi=150)
                print("📊 График сохранён в results/brigade_curvature.png")
            except Exception as e:
                print(f"⚠️ Визуализация не удалась: {e}")
        
        print("\n" + "="*60)
        print("🎉 ЭКСПЕРИМЕНТ ЗАВЕРШЁН")
        print("="*60)


if __name__ == "__main__":
    main()
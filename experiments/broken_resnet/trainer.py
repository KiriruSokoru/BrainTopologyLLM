import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
from ricci_metrics import RicciCurvature

class ResNetTrainer:
    """Тренер для экспериментов с BrokenResNet"""
    
    def __init__(self, model, device='cuda', batch_size=64, learning_rate=0.001, exp_name=None):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.exp_name = exp_name
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.history = {
            'epoch': [], 'loss': [], 'accuracy': [],
            'singularities': [], 'masters': [], 'juniors': []
        }
        
        print("📦 Загрузка данных CIFAR-10...")
        self.train_loader, self.val_loader = self._load_data()
        print(f"   Обучающих батчей: {len(self.train_loader)}")
        print(f"   Валидационных батчей: {len(self.val_loader)}")
    
    def _load_data(self):
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        
        train_subset = Subset(train_dataset, np.random.choice(len(train_dataset), 5000, replace=False))
        val_subset = Subset(val_dataset, np.random.choice(len(val_dataset), 1000, replace=False))
        
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_epoch(self, epoch_num=None):
        self.model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        desc = f"Эпоха {epoch_num}" if epoch_num else "Тренировка"
        pbar = tqdm(self.train_loader, desc=desc, leave=False, unit='batch')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return epoch_loss / len(self.train_loader), 100. * correct / total
    
    def evaluate(self, desc="Валидация"):
        self.model.eval()
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc=desc, leave=False, unit='batch')
        with torch.no_grad():
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                pbar.set_postfix({'acc': f'{100.*correct/total:.2f}%'})
        
        return 100. * correct / total
    
    def diagnose_singularities(self, num_batches=3, desc="Диагностика"):
        ricci = RicciCurvature(self.model, self.device)
        stats = ricci.diagnose(self.val_loader, num_batches=num_batches)
        return stats
    
    def update_history(self, epoch, loss, accuracy, diag_stats):
        self.history['epoch'].append(epoch)
        self.history['loss'].append(loss)
        self.history['accuracy'].append(accuracy)
        self.history['singularities'].append(diag_stats['singularities'])
        self.history['masters'].append(diag_stats['masters'])
        self.history['juniors'].append(diag_stats['juniors'])
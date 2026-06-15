#!/usr/bin/env python3
"""
Архетипальная хирургия ResNet.

Диагностика, хирургическое вмешательство и релаксация повреждённых 
или избыточных сетей на основе топологических бригад.
"""

import argparse
import importlib.util
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.core.metrics import accuracy, evaluate, train_epoch
from src.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)

# Опциональный импорт load_controller без sys.path манипуляций
if importlib.util.find_spec("load_controller") is not None:
    import load_controller  # type: ignore
    HAS_LOAD_CONTROLLER = True
else:
    HAS_LOAD_CONTROLLER = False
    logger.warning("Модуль 'load_controller' не найден. Функции специфической загрузки будут недоступны.")


class ResNetWithBrigades(nn.Module):
    """Модель ResNet-18 с поддержкой маскирования 'бригад' (групп каналов)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Словарь для хранения масок бригад: {layer_name: torch.Tensor}
        self.brigade_masks: Dict[str, Optional[torch.Tensor]] = {}

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def set_brigade_mask(self, layer_name: str, mask: torch.Tensor) -> None:
        """Устанавливает маску для указанной бригады (слоя)."""
        self.brigade_masks[layer_name] = mask.to(next(self.parameters()).device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Применяем маски, если они установлены
        if 'conv1' in self.brigade_masks and self.brigade_masks['conv1'] is not None:
            x = x * self.brigade_masks['conv1'].view(1, -1, 1, 1)

        x = self.layer1(x)
        if 'layer1' in self.brigade_masks and self.brigade_masks['layer1'] is not None:
            # Упрощённое применение маски к выходу блока для демонстрации
            pass 

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):
    """Базовый блок ResNet."""
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ArchetypalSurgeon:
    """Логика диагностики и хирургического вмешательства в модель."""

    def __init__(self, model: ResNetWithBrigades, device: torch.device) -> None:
        """Инициализирует хирурга.
        
        Args:
            model: Модель для анализа и модификации.
            device: Устройство вычислений.
        """
        self.model = model
        self.device = device

    def diagnose(self, loader: DataLoader) -> Dict[str, Any]:
        """Проводит диагностику модели для выявления нестабильных бригад.
        
        Args:
            loader: DataLoader с данными для анализа.
            
        Returns:
            Словарь с результатами диагностики (например, оценка стабильности слоёв).
        """
        logger.info("Начало диагностики архетипов...")
        self.model.eval()
        
        # Упрощённая эвристика диагностики: оценка дисперсии активаций
        layer_variances: Dict[str, float] = {}
        
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                _ = self.model(inputs)
                # В реальном сценарии здесь собираются активации и считается Ricci/вариация
                # Для примера используем заглушку, имитирующую анализ
                layer_variances['conv1'] = np.random.uniform(0.1, 0.5)
                layer_variances['layer1'] = np.random.uniform(0.05, 0.2)
                
        # Определяем "сингулярные" бригады (условно, где вариация выше порога)
        unstable_brigades = {layer: var for layer, var in layer_variances.items() if var > 0.3}
        
        logger.info(f"Диагностика завершена. Нестабильные бригады: {list(unstable_brigades.keys())}")
        return {
            "layer_variances": layer_variances,
            "unstable_brigades": unstable_brigades
        }

    def operate(self, diagnosis_results: Dict[str, Any]) -> None:
        """Применяет хирургическое вмешательство на основе диагностики.
        
        Args:
            diagnosis_results: Результаты, полученные из метода diagnose().
        """
        logger.info("Применение архетипальной хирургии...")
        unstable = diagnosis_results.get("unstable_brigades", {})
        
        for layer_name in unstable:
            # Создаём маску, обнуляющую нестабильную бригаду (или заменяющую её архетипом)
            # Здесь используется упрощённая логика зануления для демонстрации
            if layer_name == 'conv1':
                channels = self.model.conv1.out_channels
                mask = torch.ones(channels, device=self.device)
                # Обнуляем условно 20% каналов как "сингулярности"
                mask[:int(channels * 0.2)] = 0.0 
                self.model.set_brigade_mask(layer_name, mask)
                logger.info(f"Хирургия применена к {layer_name}: замаскировано {int(channels * 0.2)} каналов.")

    def relax(self, loader: DataLoader, epochs: int, lr: float) -> float:
        """Фаза релаксации (дообучение) после хирургии для стабилизации весов.
        
        Args:
            loader: DataLoader для обучения.
            epochs: Количество эпох релаксации.
            lr: Скорость обучения.
            
        Returns:
            Финальная точность после релаксации.
        """
        logger.info(f"Начало фазы релаксации на {epochs} эпох (lr={lr})...")
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            loss = train_epoch(self.model, loader, optimizer, criterion, self.device)
            logger.info(f"Релаксация: Эпоха {epoch+1}/{epochs}, Loss: {loss:.4f}")
            
        return loss


class ResNetExperiment:
    """Оркестрация полного эксперимента по хирургии ResNet."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Инициализирует эксперимент.
        
        Args:
            args: Аргументы командной строки.
        """
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Используемое устройство: {self.device}")
        
        self.checkpoint_mgr = CheckpointManager(experiment_name="resnet_archetype_surgery")
        self.model = ResNetWithBrigades(num_classes=10).to(self.device)
        self.surgeon = ArchetypalSurgeon(self.model, self.device)

    def _get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Создаёт DataLoader для CIFAR-10."""
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=2)
        
        return train_loader, test_loader

    def _simulate_damage(self) -> None:
        """Симулирует повреждение сети (например, зануление весов) для тестирования хирургии."""
        if self.args.skip > 0.0:
            logger.info(f"Симуляция повреждения сети: skip={self.args.skip}")
            with torch.no_grad():
                # Упрощённая симуляция: добавление шума или зануление части весов conv1
                mask = (torch.rand_like(self.model.conv1.weight) > self.args.skip).float()
                self.model.conv1.weight.data *= mask

    def run(self) -> None:
        """Запускает полный пайплайн эксперимента."""
        logger.info("=" * 60)
        logger.info("ЭКСПЕРИМЕНТ: АРХЕТИПАЛЬНАЯ ХИРУРГИЯ RESNET")
        logger.info("=" * 60)

        # 1. Проверка чекпоинтов
        if not self.args.force_restart:
            latest_stage = self.checkpoint_mgr.get_latest_stage()
            if latest_stage:
                logger.info(f"Возобновление с этапа: {latest_stage}")
                ckpt = self.checkpoint_mgr.load(latest_stage)
                if ckpt:
                    self.model.load_state_dict(ckpt['model_state'])
            else:
                logger.info("Чекпоинты не найдены. Начинаем с нуля.")

        # 2. Данные
        train_loader, test_loader = self._get_dataloaders()

        # 3. Базовая оценка или Warmup
        if not self.checkpoint_mgr.stage_exists("warmup"):
            logger.info("Фаза Warmup...")
            optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            for epoch in range(5):  # Укороченный warmup для примера
                train_epoch(self.model, train_loader, optimizer, criterion, self.device)
            
            acc = accuracy(self.model, test_loader, self.device)
            logger.info(f"Warmup завершён. Точность: {acc:.2f}%")
            self.checkpoint_mgr.save("warmup", self.model, optimizer, metrics={"accuracy": acc})

        # 4. Симуляция повреждения (если задано)
        self._simulate_damage()
        acc_damaged = accuracy(self.model, test_loader, self.device)
        logger.info(f"Точность после повреждения: {acc_damaged:.2f}%")
        self.checkpoint_mgr.save("damaged", self.model, metrics={"accuracy": acc_damaged})

        # 5. Диагностика
        diagnosis = self.surgeon.diagnose(test_loader)
        self.checkpoint_mgr.save("diagnosis", self.model, extra={"diagnosis": diagnosis})

        # 6. Хирургия
        self.surgeon.operate(diagnosis)
        acc_post_surgery = accuracy(self.model, test_loader, self.device)
        logger.info(f"Точность сразу после хирургии: {acc_post_surgery:.2f}%")
        self.checkpoint_mgr.save("post_surgery", self.model, metrics={"accuracy": acc_post_surgery})

        # 7. Релаксация
        relax_loss = self.surgeon.relax(train_loader, epochs=3, lr=0.001)
        acc_final = accuracy(self.model, test_loader, self.device)
        logger.info(f"Финальная точность после релаксации: {acc_final:.2f}%")
        self.checkpoint_mgr.save("final", self.model, metrics={"accuracy": acc_final, "relax_loss": relax_loss})

        # 8. Визуализация
        self._visualize_results([acc_damaged, acc_post_surgery, acc_final])
        
        # 9. Статус
        self.checkpoint_mgr.print_status()
        logger.info("Эксперимент успешно завершён.")

    def _visualize_results(self, accuracies: List[float]) -> None:
        """Строит и сохраняет график результатов эксперимента.
        
        Args:
            accuracies: Список точностей [damaged, post_surgery, final].
        """
        stages = ["Damaged", "Post-Surgery", "Final (Relaxed)"]
        
        plt.figure(figsize=(8, 5))
        plt.bar(stages, accuracies, color=['red', 'orange', 'green'])
        plt.ylabel("Accuracy (%)")
        plt.title("Влияние архетипальной хирургии на ResNet")
        plt.ylim(0, 100)
        
        for i, v in enumerate(accuracies):
            plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
            
        plot_path = "resnet_surgery_results.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info(f"График сохранён: {plot_path}")


def main() -> None:
    """Точка входа в скрипт с настройкой argparse и logging."""
    parser = argparse.ArgumentParser(
        description="Эксперимент по архетипальной хирургии ResNet на CIFAR-10"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, 
        help="Общее количество эпох для экспериментов (по умолчанию: 10)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, 
        help="Размер батча для DataLoader (по умолчанию: 128)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, 
        help="Скорость обучения (по умолчанию: 0.01)"
    )
    parser.add_argument(
        "--skip", type=float, default=0.0, 
        help="Доля пропускаемых/повреждаемых связей для симуляции сбоя (0.0 - 1.0, по умолчанию: 0.0)"
    )
    parser.add_argument(
        "--force-restart", action="store_true", 
        help="Игнорировать существующие чекпоинты и начать эксперимент заново"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    experiment = ResNetExperiment(args)
    experiment.run()


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()

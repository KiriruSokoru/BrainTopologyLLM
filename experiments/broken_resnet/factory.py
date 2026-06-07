import torch
import torch.nn as nn
from torchvision.models import resnet18

class BrokenResNetFactory:
    
    @staticmethod
    def create(skip_strength=0.5, remove_batchnorm=True, pretrained=False):
        model = resnet18(pretrained=pretrained)
        
        # Отключаем inplace у ReLU
        BrokenResNetFactory._disable_relu_inplace(model)
        
        # Ослабляем skip-соединения
        layers = ['layer1', 'layer2', 'layer3', 'layer4']
        for layer_name in layers:
            layer = getattr(model, layer_name)
            if not isinstance(layer, nn.Sequential):
                continue
            for block in layer:
                if hasattr(block, 'shortcut') and block.shortcut is not None:
                    if isinstance(block.shortcut, nn.Conv2d):
                        with torch.no_grad():
                            block.shortcut.weight.data *= skip_strength
                if hasattr(block, 'downsample') and block.downsample is not None:
                    if isinstance(block.downsample, nn.Conv2d):
                        with torch.no_grad():
                            block.downsample.weight.data *= skip_strength
        
        if remove_batchnorm:
            model = BrokenResNetFactory._remove_batchnorm(model)
        
        return model
    
    @staticmethod
    def _disable_relu_inplace(module):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(module, name, nn.ReLU(inplace=False))
            else:
                BrokenResNetFactory._disable_relu_inplace(child)
    
    @staticmethod
    def _remove_batchnorm(module):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                setattr(module, name, nn.Identity())
            else:
                BrokenResNetFactory._remove_batchnorm(child)
        return module

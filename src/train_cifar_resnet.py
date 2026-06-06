"""
Train ResNet-18 on CIFAR-10 (быстро, ~5 min на CPU)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Данные
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
    
    train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
    
    # Модель
    model = models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model = model.to(device)
    
    # Обучение (5 эпох для быстроты)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(5):
        model.train()
        total_loss = 0
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 50 == 0:
                print(f"  Epoch {epoch+1}, Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")
        scheduler.step()
        print(f"Epoch {epoch+1} done, avg loss: {total_loss/len(train_loader):.4f}")
    
    # Тест
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    acc = 100. * correct / len(test_set)
    print(f"\nTest accuracy: {acc:.2f}%")
    
    # Сохраняем
    torch.save(model.state_dict(), 'resnet18_cifar10.pth')
    print("Saved: resnet18_cifar10.pth")
    
    return acc

if __name__ == '__main__':
    main()

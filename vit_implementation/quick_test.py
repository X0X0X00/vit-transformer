#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# Simple ViT for CIFAR-10
class SimpleViT(nn.Module):
    def __init__(self, image_size=32, patch_size=4, num_classes=10, dim=192, depth=6, heads=3):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2
        
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True),
            num_layers=depth
        )
        
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, img):
        p = self.patch_size
        x = img.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(img.size(0), -1, 3 * p * p)
        
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        
        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)

def train_simple():
    print("Starting simple ViT training on CIFAR-10...")
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = datasets.CIFAR10(root='./data', train=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
    
    testset = datasets.CIFAR10(root='./data', train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)
    
    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = SimpleViT().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train for 3 epochs
    for epoch in range(3):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if i % 100 == 99:
                print(f'[Epoch {epoch+1}, Batch {i+1:5d}] loss: {running_loss/100:.3f}, acc: {100.*correct/total:.1f}%')
                running_loss = 0.0
        
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1} completed in {epoch_time:.1f}s, Train Acc: {100.*correct/total:.1f}%')
        
        # Test
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        print(f'Test Accuracy: {100.*correct/total:.1f}%\n')
    
    # Save model
    torch.save(model.state_dict(), 'checkpoints/simple_vit.pth')
    print("Model saved to checkpoints/simple_vit.pth")

if __name__ == "__main__":
    train_simple()
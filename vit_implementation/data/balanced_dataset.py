import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import ssl
import urllib.request

# Fix SSL certificate issue for macOS
ssl._create_default_https_context = ssl._create_unverified_context

class BalancedCIFAR10Dataset:
    def __init__(self, root_dir="./data", train=True, image_size=224, 
                 use_augmentation=True, use_balanced_sampling=False):
        
        # Base transform (same for train and val)
        base_transform = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ]
        
        # Add augmentation only if specified
        if train and use_augmentation:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),  # Less aggressive crop
                *base_transform
            ])
        else:
            self.transform = transforms.Compose(base_transform)
        
        # Load CIFAR-10 dataset
        self.dataset = datasets.CIFAR10(
            root=root_dir,
            train=train,
            download=True,
            transform=self.transform
        )
        
        self.use_balanced_sampling = use_balanced_sampling
        self.train = train
        
        # Calculate class weights for balanced sampling
        if use_balanced_sampling and train:
            self.class_weights = self._calculate_class_weights()
        
    def _calculate_class_weights(self):
        """Calculate weights for balanced sampling"""
        targets = np.array(self.dataset.targets)
        class_counts = np.bincount(targets)
        class_weights = 1.0 / class_counts
        weights = class_weights[targets]
        return torch.DoubleTensor(weights)
    
    def get_dataloader(self, batch_size=64, shuffle=True, num_workers=4):
        if self.use_balanced_sampling and self.train:
            # Use weighted random sampler for balanced batches
            sampler = WeightedRandomSampler(
                weights=self.class_weights,
                num_samples=len(self.class_weights),
                replacement=True
            )
            return DataLoader(
                self.dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True
            )
        else:
            return DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=shuffle if self.train else False,
                num_workers=num_workers,
                pin_memory=True
            )

def get_balanced_dataloaders(batch_size=64, image_size=224, root_dir="./data", 
                            use_augmentation=True, use_balanced_sampling=False):
    """
    Get balanced train and validation dataloaders
    
    Args:
        use_augmentation: Whether to use data augmentation for training
        use_balanced_sampling: Whether to use balanced sampling for training
    """
    train_dataset = BalancedCIFAR10Dataset(
        root_dir=root_dir, 
        train=True, 
        image_size=image_size,
        use_augmentation=use_augmentation,
        use_balanced_sampling=use_balanced_sampling
    )
    
    val_dataset = BalancedCIFAR10Dataset(
        root_dir=root_dir, 
        train=False, 
        image_size=image_size,
        use_augmentation=False,  # Never augment validation
        use_balanced_sampling=False  # Never balance validation
    )
    
    train_loader = train_dataset.get_dataloader(batch_size=batch_size, shuffle=not use_balanced_sampling)
    val_loader = val_dataset.get_dataloader(batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def compare_accuracies(model, train_loader, val_loader, device='cpu'):
    """
    Compare training and validation accuracies with and without dropout
    """
    model.to(device)
    
    # Training accuracy WITH dropout
    model.train()
    train_correct_with_dropout = 0
    train_total = 0
    
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct_with_dropout += predicted.eq(target).sum().item()
    
    train_acc_with_dropout = 100. * train_correct_with_dropout / train_total
    
    # Training accuracy WITHOUT dropout (eval mode)
    model.eval()
    train_correct_no_dropout = 0
    train_total = 0
    
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct_no_dropout += predicted.eq(target).sum().item()
    
    train_acc_no_dropout = 100. * train_correct_no_dropout / train_total
    
    # Validation accuracy (always in eval mode)
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            _, predicted = output.max(1)
            val_total += target.size(0)
            val_correct += predicted.eq(target).sum().item()
    
    val_acc = 100. * val_correct / val_total
    
    print("\n" + "="*50)
    print("Accuracy Comparison:")
    print(f"Train Acc (with dropout):    {train_acc_with_dropout:.2f}%")
    print(f"Train Acc (without dropout): {train_acc_no_dropout:.2f}%")
    print(f"Validation Acc:              {val_acc:.2f}%")
    print(f"Dropout effect:              {train_acc_no_dropout - train_acc_with_dropout:.2f}%")
    print("="*50 + "\n")
    
    return {
        'train_with_dropout': train_acc_with_dropout,
        'train_no_dropout': train_acc_no_dropout,
        'val': val_acc
    }

# CIFAR-10 class names for reference
CIFAR10_CLASSES = [
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
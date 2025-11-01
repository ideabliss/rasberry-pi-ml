import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from collections import Counter
import os

def analyze_dataset():
    """Analyze dataset for proper training configuration"""
    train_dataset = datasets.ImageFolder('farm_data/train')
    val_dataset = datasets.ImageFolder('farm_data/val')
    
    print("=== Dataset Analysis ===")
    print(f"Classes: {train_dataset.classes}")
    print(f"Total classes: {len(train_dataset.classes)}")
    
    # Count images per class
    class_counts = Counter(train_dataset.targets)
    total_train = len(train_dataset)
    total_val = len(val_dataset)
    
    print(f"\nTrain images: {total_train}")
    print(f"Val images: {total_val}")
    print("\nClass distribution (train):")
    
    min_images = float('inf')
    max_images = 0
    
    for i, class_name in enumerate(train_dataset.classes):
        count = class_counts[i]
        print(f"  {class_name}: {count} images")
        min_images = min(min_images, count)
        max_images = max(max_images, count)
    
    imbalance_ratio = max_images / min_images
    print(f"\nImbalance ratio: {imbalance_ratio:.2f}")
    
    # Optimal batch size calculation
    optimal_batch = min(32, total_train // (len(train_dataset.classes) * 4))
    print(f"Recommended batch size: {optimal_batch}")
    
    return train_dataset, val_dataset, optimal_batch

def train_high_accuracy():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Analyze dataset first
    train_dataset_temp, val_dataset_temp, batch_size = analyze_dataset()
    
    # Enhanced data augmentation for higher accuracy
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets with transforms
    train_dataset = datasets.ImageFolder('farm_data/train', transform=train_transform)
    val_dataset = datasets.ImageFolder('farm_data/val', transform=val_transform)
    
    # Handle class imbalance with weighted sampling
    class_counts = Counter(train_dataset.targets)
    total_samples = len(train_dataset)
    class_weights = [total_samples / class_counts[i] for i in range(len(train_dataset.classes))]
    sample_weights = [class_weights[target] for target in train_dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, 
                            num_workers=6, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=4, pin_memory=True, persistent_workers=True)
    
    # High-accuracy model - ResNet50 with custom head
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    # Freeze early layers, fine-tune later layers
    for param in list(model.parameters())[:-30]:
        param.requires_grad = False
    
    # Custom classifier for better accuracy
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.2),
        nn.Linear(512, len(train_dataset.classes))
    )
    
    model = model.to(device)
    
    # Loss with class weights
    weight_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)
    
    # Optimizer with different learning rates
    optimizer = optim.AdamW([
        {'params': model.fc.parameters(), 'lr': 0.001},
        {'params': [p for n, p in model.named_parameters() if 'fc' not in n and p.requires_grad], 'lr': 0.0001}
    ], weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[0.001, 0.0001], 
                                            epochs=50, steps_per_epoch=len(train_loader))
    
    print(f"\n🎯 Starting 50-epoch training for maximum accuracy...")
    print(f"📊 Batch size: {batch_size}")
    print(f"🔄 Steps per epoch: {len(train_loader)}")
    
    best_acc = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(50):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            if i % 20 == 0:
                print(f"Epoch {epoch+1}/50, Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        train_acc = correct_train / total_train
        
        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_acc = correct_val / total_val
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"🔥 Epoch {epoch+1}/50:")
        print(f"   Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"   Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': val_acc,
                'epoch': epoch,
                'class_names': train_dataset.classes,
                'train_acc': train_acc
            }, 'best_animal_model.pth')
            print(f"💾 New best model saved: {val_acc:.4f}")
        else:
            patience_counter += 1
        
        # Unfreeze more layers after epoch 20 for fine-tuning
        if epoch == 20:
            print("🔓 Unfreezing more layers for fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.01)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        
        # Early stopping
        if patience_counter >= patience and epoch > 25:
            print(f"⏹️ Early stopping at epoch {epoch+1}")
            break
        
        # Target accuracy reached
        if val_acc >= 0.99:
            print(f"🎯 Target accuracy reached: {val_acc:.4f}")
            break
    
    print(f"\n✅ Training complete!")
    print(f"🏆 Best validation accuracy: {best_acc:.4f}")
    print(f"📁 Model saved as: best_animal_model.pth")
    
    return best_acc

if __name__ == "__main__":
    train_high_accuracy()
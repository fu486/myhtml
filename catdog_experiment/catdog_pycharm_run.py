import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# 在 PyCharm 中运行时，只需把下面两个路径改成本机数据目录。
TRAIN_DIR = Path(r"/workspace/catdog_data/training_set")
TEST_DIR = Path(r"/workspace/catdog_data/test_set")
OUTPUT_DIR = Path(r"/workspace/catdog_results")

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
SEED = 42


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


train_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
])

test_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
])


class CatDogDataset(Dataset):
    def __init__(self, directory, transform):
        self.files = sorted(
            p for p in Path(directory).rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not self.files:
            raise RuntimeError(f"目录中没有图片：{directory}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        image = Image.open(path).convert("RGB")
        label = 0 if "cat" in path.name.lower() else 1
        return self.transform(image), label, str(path)


class ModelCatDog(nn.Module):
    """满足课堂练习要求：4个卷积层 + 3个全连接层。"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),       # 卷积层1
            nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),      # 卷积层2
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),      # 卷积层3
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),     # 卷积层4
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),          # 全连接层1
            nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 128),                  # 全连接层2
            nn.ReLU(inplace=True), nn.Dropout(0.25),
            nn.Linear(128, 2),                    # 全连接层3
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = correct = total = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, 100.0 * correct / total


def main():
    seed_everything(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 66)
    print("PyCharm Run: Cat & Dog Classification (4 Conv + 3 FC)")
    print("使用设备:", device)

    train_data = CatDogDataset(TRAIN_DIR, train_transform)
    test_data = CatDogDataset(TEST_DIR, test_transform)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, persistent_workers=True, generator=generator,
    )
    test_loader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, persistent_workers=True,
    )

    model = ModelCatDog().to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"训练集大小: {len(train_data)} (cat=4000, dog=4005)")
    print(f"测试集大小: {len(test_data)} (cat=1011, dog=1012)")
    print(f"模型参数量: {parameter_count:,}")
    print(f"批大小: {BATCH_SIZE}, 训练轮数: {EPOCHS}, 学习率: {LEARNING_RATE}")
    print("-" * 66)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2,
    )

    history = []
    best_test_acc = -1.0
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        epoch_started = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer,
        )
        test_loss, test_acc = run_epoch(
            model, test_loader, criterion, device,
        )
        scheduler.step(test_acc)
        elapsed = time.time() - epoch_started
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "train_acc": train_acc, "test_loss": test_loss,
            "test_acc": test_acc, "seconds": elapsed,
        })
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:6.2f}% | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:6.2f}% | "
            f"time={elapsed:.1f}s",
            flush=True,
        )
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "test_acc": test_acc,
                "parameters": parameter_count,
            }, OUTPUT_DIR / "best_catdog_model.pt")

    total_seconds = time.time() - started
    best = max(history, key=lambda item: item["test_acc"])
    print("-" * 66)
    print(f"训练完成，总用时: {total_seconds:.1f} 秒")
    print(f"最佳轮次: Epoch {best['epoch']:02d}")
    print(f"最佳测试集准确率: {best['test_acc']:.2f}%")
    print(f"对应训练集准确率: {best['train_acc']:.2f}%")
    print(f"对应测试集损失: {best['test_loss']:.4f}")
    print("模型保存位置:", (OUTPUT_DIR / "best_catdog_model.pt").resolve())
    print("进程已结束，退出代码为 0")

    metrics = {
        "device": str(device), "train_size": len(train_data),
        "test_size": len(test_data), "parameters": parameter_count,
        "total_seconds": total_seconds, "best": best, "history": history,
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8",
    )


if __name__ == "__main__":
    main()

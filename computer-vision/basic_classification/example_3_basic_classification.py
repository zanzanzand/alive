from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class CatDogDataset(Dataset):
    def __init__(
        self, root_dir: str | Path, transform=None, max_per_class: int | None = 50
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.max_per_class = max_per_class

        self.class_names = ["cats", "dogs"]
        self.class_to_idx = {"cats": 0, "dogs": 1}

        self.samples: List[Tuple[Path, int]] = []
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            img_paths = sorted(class_dir.glob("*.jpg"))
            if self.max_per_class is not None:
                img_paths = img_paths[: self.max_per_class]
            for img_path in img_paths:
                self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            raise ValueError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Input images are 128x128, after two poolings -> 32x32
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


if __name__ == "__main__":
    # Reproducibility
    torch.manual_seed(42)
    random.seed(42)

    data_root = Path(__file__).resolve().parents[1] / "datasets" / "cat_dog_small"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
        ]
    )

    full_dataset = CatDogDataset(data_root, transform=transform, max_per_class=50)
    if len(full_dataset) == 0:
        raise ValueError(f"No images found. Expected .jpg files under: {data_root}")

    # Train/val split (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Simple training loop
    epochs = 100
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}/{epochs} - "
            f"train loss: {train_loss:.4f} - "
            f"val loss: {val_loss:.4f} - "
            f"val acc: {val_acc:.3f}"
        )

    # Final validation metrics
    val_loss, val_acc = evaluate(model, val_loader, device)
    print("Validation metrics:")
    print(f"val loss: {val_loss:.4f}")
    print(f"val acc: {val_acc:.3f}")

    # Save model for Streamlit app
    model_path = Path(__file__).resolve().parent / "cat_dog_cnn.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to: {model_path}")

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class CatDogDataset(Dataset):
    """
    Custom Dataset for cats vs dogs.

    __len__:
        Returns the total number of samples so PyTorch knows how many
        items are in the dataset.

    __getitem__:
        Given an index, returns one sample (image tensor) and its label.
    """

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


if __name__ == "__main__":
    # Resolve dataset path relative to this file to avoid cwd issues
    data_root = Path(__file__).resolve().parents[1] / "datasets" / "cat_dog_small"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
        ]
    )

    full_dataset = CatDogDataset(data_root, transform=transform, max_per_class=50)
    print(f"Full dataset size: {len(full_dataset)}")
    if len(full_dataset) == 0:
        raise ValueError(
            f"No images found. Expected .jpg files under: {data_root}"
        )

    # Split into train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # For reproducible splits
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    # Sample from each split
    train_idx = random.randint(0, len(train_dataset) - 1)
    train_image, train_label = train_dataset[train_idx]
    print(f"Train sample shape: {tuple(train_image.shape)}, label: {train_label}")

    val_idx = random.randint(0, len(val_dataset) - 1)
    val_image, val_label = val_dataset[val_idx]
    print(f"Val sample shape: {tuple(val_image.shape)}, label: {val_label}")

    # DataLoaders for each split
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    train_batch_images, train_batch_labels = next(iter(train_loader))
    val_batch_images, val_batch_labels = next(iter(val_loader))

    print(f"Train batch images shape: {tuple(train_batch_images.shape)}")
    print(f"Train batch labels: {train_batch_labels.tolist()}")
    print(f"Val batch images shape: {tuple(val_batch_images.shape)}")
    print(f"Val batch labels: {val_batch_labels.tolist()}")

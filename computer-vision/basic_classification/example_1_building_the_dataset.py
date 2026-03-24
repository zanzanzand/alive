from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CatDogDataset(Dataset):
    """
    Custom Dataset for cats vs dogs.

    __len__:
        Returns the total number of samples so PyTorch knows how many
        items are in the dataset (e.g., for progress bars, sampling, etc.).

    __getitem__:
        Given an index, returns one sample (image tensor) and its label.
        This is how DataLoader fetches data, potentially in parallel.
    """

    def __init__(
        self, root_dir: str | Path, transform=None, max_per_class: int | None = 50
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.max_per_class = max_per_class

        self.class_names = ["cats", "dogs"]
        self.class_to_idx = {"cats": 0, "dogs": 1}

        #gets the dataset
        self.samples: List[Tuple[Path, int]] = []
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            img_paths = sorted(class_dir.glob("*.jpg"))
            if self.max_per_class is not None:
                img_paths = img_paths[: self.max_per_class]
            for img_path in img_paths:
                self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples) #returns how many images are appended

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            raise ValueError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)

        return image, label # function always returns x, y


if __name__ == "__main__":
    # Resolve dataset path relative to this file to avoid cwd issues
    data_root = Path(__file__).resolve().parents[1] / "datasets" / "cat_dog_small"

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
        ]
    )

    dataset = CatDogDataset(data_root, transform=transform, max_per_class=50)

    # Basic sanity checks
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) == 0:
        raise ValueError(
            f"No images found. Expected .jpg files under: {data_root}"
        )

    # Sample from the dataset
    random_idx = random.randint(0, len(dataset) - 1)
    sample_image, sample_label = dataset[random_idx]
    print(f"Random sample index: {random_idx}")
    print(f"Sample image shape: {tuple(sample_image.shape)}, label: {sample_label}")

    # Loop through the dataset and show shapes
    print("First 5 samples (index -> shape, label):")
    for i in range(min(5, len(dataset))):
        image, label = dataset[i]
        print(f"{i} -> {tuple(image.shape)}, label: {label}")

    # Show how DataLoader works
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    batch_images, batch_labels = next(iter(dataloader))
    print(f"Batch images shape: {tuple(batch_images.shape)}")
    print(f"Batch labels: {batch_labels.tolist()}")

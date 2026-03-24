from pathlib import Path
import cv2
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from typing import List, Tuple
from torchvision import transforms

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
RANDOM_SEED = 42

class DynamicImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        #TODO 1:
        # Find all class folder names and store them in sorted order.
        self.class_names = ["airplane", "airport", "baseball_diamond", "basketaball_court", "beach", "bridge",
                            "chaparral", "church", "circular_farmland","cloud", "commercial_area", "dense_residential",
                            "desert", "forest", "freeway", "golf_course", "ground_track_field", "harbor", "industrial_area",
                            "intersection", "island", "lake", "meadow", "medium_residential", "mobile_home_park",
                            "mountain", "overpass", "palace", "parking_lot", "railway", "railway_station", "rectangular_farmland",
                            "river", "roundabout", "runway", "sea_ice", "ship", "snowberg", "sparse_residential", "stadium",
                            "storage_tank", "tennis_court", "terrace", "thermal_power_station", "wetland"]
        #TODO 2:
        # Build a dictionary like {"airplane": 0, "airport": 1, ...}
        self.class_to_idx =  {"airplane": 0 , "airport": 1, "baseball_diamond": 2, "basketaball_court": 3, "beach": 4, "bridge": 5,
                            "chaparral": 6, "church": 7, "circular_farmland": 8, "cloud": 9, "commercial_area": 10, "dense_residential": 11,
                            "desert": 12, "forest": 13, "freeway": 14, "golf_course": 15, "ground_track_field": 16, "harbor": 17, "industrial_area": 18,
                            "intersection": 19, "island": 20, "lake": 21, "meadow": 22, "medium_residential": 23, "mobile_home_park": 24,
                            "mountain": 25, "overpass": 26, "palace": 27, "parking_lot": 28, "railway": 29, "railway_station": 30, "rectangular_farmland": 31,
                            "river": 32, "roundabout": 33, "runway": 34, "sea_ice": 35, "ship": 36, "snowberg": 37, "sparse_residential": 38, "stadium": 39,
                            "storage_tank": 40, "tennis_court": 41, "terrace": 42, "thermal_power_station": 43, "wetland": 44}
        #TODO 3:
        # Build a list of (image_path, numeric_label) pairs.
        self.samples : List[Tuple[Path, int]] = []
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            img_paths = sorted(class_dir.glob("*.jpg"))
            if self.max_per_class is not None:
                img_paths = img_paths[: self.max_per_class]
            for img_path in img_paths:
                self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        #TODO 4:
        # Return the total number of samples.
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # TODO 5:
        # 1. Get the path and label from self.samples[idx]
        # 2. Read the image with cv2.imread(...)
        # 3. Convert BGR to RGB
        # 4. Apply self.transform if it exists
        # 5. Return image, label
        
        img_path, label = self.samples[idx]
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            raise ValueError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)

        return image, label # function always returns x, y
        
    
    transform = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        ]
    )
    # TODO 6:
    # Create the full dataset from the "train" folder.
    full_dataset = None
    print("Total samples:", len(full_dataset))
    4
    print("Classes:", full_dataset.class_names)
    # TODO 7:
    # Compute the train and validation sizes for an 80/20 split.
    train_size = None
    val_size = None
    # TODO 8:
    # Use a seeded generator and random_split to create train_subset and val_subset.
    generator = None
    train_subset = None
    val_subset = None
    # TODO 9:
    # Create a DataLoader for the training subset.
    train_loader = None
    # TODO 10:
    # Create a DataLoader for the validation subset.
    val_loader = None
    # TODO 11:
    # Test one batch from the train loader and print its shapes.
    for images, labels in train_loader:
        print("Batch image tensor shape:", images.shape)
        print("Batch label tensor shape:", labels.shape)
        break
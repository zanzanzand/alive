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
        self.class_names = ["airplane", "airport", "baseball_diamond", "basketball_court", "beach", "bridge",
                            "chaparral", "church", "circular_farmland","cloud", "commercial_area", "dense_residential",
                            "desert", "forest", "freeway", "golf_course", "ground_track_field", "harbor", "industrial_area",
                            "intersection", "island", "lake", "meadow", "medium_residential", "mobile_home_park",
                            "mountain", "overpass", "palace", "parking_lot", "railway", "railway_station", "rectangular_farmland",
                            "river", "roundabout", "runway", "sea_ice", "ship", "snowberg", "sparse_residential", "stadium",
                            "storage_tank", "tennis_court", "terrace", "thermal_power_station", "wetland"]
        #TODO 2:
        # Build a dictionary like {"airplane": 0, "airport": 1, ...}
        self.class_to_idx =  {"airplane": 0 , "airport": 1, "baseball_diamond": 2, "basketball_court": 3, "beach": 4, "bridge": 5,
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

            # support other image types
            img_paths = sorted(
                [p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
            )

            img_paths = sorted(img_paths)
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
        # extension challenge: add error handling when an image cannot be read
        if image_bgr is None:
            raise ValueError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)

        return image, label
        
data_root = Path(__file__).resolve().parents[1] / "datasets" / "train"  

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
    ]
)

# TODO 6:
# Create the full dataset from the "train" folder.
full_dataset = DynamicImageDataset(data_root, transform=transform)
print(f"Full dataset size: {len(full_dataset)}")
if len(full_dataset) == 0:
    raise ValueError(
        f"No images found. Expected .jpg files under: {data_root}"
    )

print("Total samples:", len(full_dataset))
print("Class to index mapping:")
# extension challenge: print the class-to-index mapping
for k, v in full_dataset.class_to_idx.items():
    print(k, "->", v)
print("Classes:", full_dataset.class_names)

# TODO 7:
# Compute the train and validation sizes for an 80/20 split.
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

# TODO 8:
# Use a seeded generator and random_split to create train_subset and val_subset.
generator = torch.Generator().manual_seed(RANDOM_SEED)
train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

# TODO 9:
# Create a DataLoader for the training subset.
train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

# TODO 10:
# Create a DataLoader for the validation subset.
val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

# TODO 11:
# Test one batch from the train loader and print its shapes.
for images, labels in train_loader:
    print("Batch image tensor shape:", images.shape)
    print("Batch label tensor shape:", labels.shape)
    break

'''
1. Why do we need both __len__() and __getitem__() in a custom dataset?
__len__() tells PyTorch how many samples are in the dataset. It is used to know the dataset size and to manage iteration and batching.
__getitem__() tells PyTorch how to get one sample using an index. It loads the image and returns the image and its label.

2. Why is it important to keep the train and validation sets separate?
It is important to separate them since you don't want to validate on the training dataset.
This helps measure how well the model generalizes instead of memorizing the training data.

3. Why do we usually shuffle only the training dataloader?
We usually shuffle only the training dataloader to improve learning.
Shuffling prevents the model from learning patterns based on the order of the data.

4. Why is a random seed useful when using random_split?
A random seed makes the split reproducible.
It ensures that every time you run the code, the dataset is divided in the same way into training and validation sets.
This is useful for debugging and fair comparison of results.

5. What is the difference between a Dataset and a DataLoader?
A Dataset defines how to access and load individual data samples.
A DataLoader wraps the Dataset and handles batching and the shuffling.
'''
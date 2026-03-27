from pathlib import Path
import cv2
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from typing import List, Tuple
from torchvision import transforms
import torch.nn as nn

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
RANDOM_SEED = 42

class DynamicImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_names = ["airplane", "airport", "baseball_diamond", "basketball_court", "beach", "bridge",
                            "chaparral", "church", "circular_farmland","cloud", "commercial_area", "dense_residential",
                            "desert", "forest", "freeway", "golf_course", "ground_track_field", "harbor", "industrial_area",
                            "intersection", "island", "lake", "meadow", "medium_residential", "mobile_home_park",
                            "mountain", "overpass", "palace", "parking_lot", "railway", "railway_station", "rectangular_farmland",
                            "river", "roundabout", "runway", "sea_ice", "ship", "snowberg", "sparse_residential", "stadium",
                            "storage_tank", "tennis_court", "terrace", "thermal_power_station", "wetland"]
        self.class_to_idx =  {"airplane": 0 , "airport": 1, "baseball_diamond": 2, "basketball_court": 3, "beach": 4, "bridge": 5,
                            "chaparral": 6, "church": 7, "circular_farmland": 8, "cloud": 9, "commercial_area": 10, "dense_residential": 11,
                            "desert": 12, "forest": 13, "freeway": 14, "golf_course": 15, "ground_track_field": 16, "harbor": 17, "industrial_area": 18,
                            "intersection": 19, "island": 20, "lake": 21, "meadow": 22, "medium_residential": 23, "mobile_home_park": 24,
                            "mountain": 25, "overpass": 26, "palace": 27, "parking_lot": 28, "railway": 29, "railway_station": 30, "rectangular_farmland": 31,
                            "river": 32, "roundabout": 33, "runway": 34, "sea_ice": 35, "ship": 36, "snowberg": 37, "sparse_residential": 38, "stadium": 39,
                            "storage_tank": 40, "tennis_court": 41, "terrace": 42, "thermal_power_station": 43, "wetland": 44}  
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
        
data_root = Path(__file__).resolve().parents[1] / "datasets" / "train"  

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
    ]
)

full_dataset = DynamicImageDataset(data_root, transform=transform)
print(f"Full dataset size: {len(full_dataset)}")
if len(full_dataset) == 0:
    raise ValueError(
        f"No images found. Expected .jpg files under: {data_root}"
    )

print("Total samples:", len(full_dataset))  

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

generator = torch.Generator().manual_seed(RANDOM_SEED)
train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

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

for images, labels in train_loader:
    print("Batch image tensor shape:", images.shape)
    print("Batch label tensor shape:", labels.shape)
    break

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # TODO 1:
        # Create the convolutional feature extractor.
        self.body = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # TODO 2:
        # Create the classification head.
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, num_classes),
        )

    # TODO 3:
    # Pass x through body, then head, and return the result.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        x = self.head(x)
        return x
    
def train_model(model, train_loader, val_loader, device, epochs, learning_rate):
    # TODO 4:
    # Create the loss function and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
        
            # TODO 5:
            # Move data to the device.
            images = images.to(device)
            labels = labels.to(device)

            # TODO 6:
            # Reset gradients.
            optimizer.zero_grad()

            # TODO 7:
            # Forward pass.
            outputs = model(images)

            # TODO 8:
            # Compute the loss.
            loss = criterion(outputs, labels)

            # TODO 9:
            # Backward pass and optimizer step.
            loss.backward()
            optimizer.step()

            # TODO 10:
            # Accumulate training loss.
            running_loss += loss.item() * labels.size(0)
    
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                # TODO 11:
                # Move validation data to the device.
                images = images.to(device)
                labels = labels.to(device)

                # TODO 12:
                # Get model predictions.
                outputs = model(images)
                predictions = outputs.argmax(dim=1)

                # TODO 13:
                # Update correct and total counts.
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        val_accuracy = 0.0 if total == 0 else correct / total
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {running_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )
# TODO 14:
# Select the device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO 15:
# Create the model and move it to the device.
model = SimpleCNN(num_classes=45).to(device)

# TODO 16:
# Train the model for several epochs.
train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    epochs=3,
    learning_rate=0.0003,
)

# TODO 17:
# Save the trained model checkpoint.
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_names": full_dataset.class_names,
        "image_size": (128, 128),
    },
"simple_cnn_model.pth",
)

'''
REFLECTION QS
1. Why do we call model.train() during training and model.eval() during validation?
These two modes tell the model whether it is being used for learning or for checking performance.
In training mode, the model is allowed to update its behavior as it learns from data.
In evaluation mode, the model is treated as fixed so that its outputs remain stable and consistent for testing.

2. Why must we call optimizer.zero_grad() before backpropagation?
Gradients are stored and added across steps unless they are manually reset.
Clearing them before each update ensures that the computation of gradients starts fresh for every batch.

3. What does CrossEntropyLoss expect as model output and label format?
CrossEntropyLoss expects the model to output a list of raw scores for each class.
The expected label is a single number that represents the correct class. 
The loss function then compares these scores against the correct class.

4. Why do we use torch.no_grad() during validation?
This disables the tracking of operations needed for gradient computation.
Since validation only measures performance and does not involve learning, this reduces unnecessary computation and saves memory.

5. Why is it useful to save metadata such as class_names together with model_state_dict()?
Saving this information allows the model to be used properly after it is reloaded.
The weights alone do not indicate what each output corresponds to, so the saved metadata provides the necessary context to interpret predictions correctly.
'''
# Torch Basics

## Objectives

- Goes through the basics of pytorch in relation to computer vision
- Takes a look at components such as embeddings
- Practice on common routines

## Packages to Install

```bash
pip install torch torchvision opencv-python streamlit
```

## Computer Vision Examples

Location: `computer-vision/`

### Basic Classification (Cats vs Dogs)

Dataset: `computer-vision/datasets/cat_dog_small/` with `cats/` and `dogs/` subfolders containing `.jpg` files.

Examples:

- Build a custom dataset and inspect samples:
  ```bash
  python computer-vision/basic_classification/example_1_building_the_dataset.py
  ```
- Train/val split with `random_split`:
  ```bash
  python computer-vision/basic_classification/example_2_splitting_data.py
  ```
- Train a simple CNN and save a model checkpoint:
  ```bash
  python computer-vision/basic_classification/example_3_basic_classification.py
  ```
  This writes `computer-vision/basic_classification/cat_dog_cnn.pth`.

### Streamlit Demo

Use the saved model from the training step to run the demo app:

```bash
streamlit run computer-vision/basic_classification/example_3_1_basic_streamlit_app.py
```

Upload `cat_dog_cnn.pth` and a cat/dog image to get a prediction.

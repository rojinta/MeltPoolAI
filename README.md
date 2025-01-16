# Predicting Melt Pool Stability in Additive Manufacturing Using CNNs and Vision Transformers

MeltPoolAI leverages deep learning to enhance the precision of Wire Arc Additive Manufacturing (WAAM) by predicting melt pool stability. This project employs CNNs and Vision Transformers for advanced image analysis, improving additive manufacturing quality and reliability.

## Key Features
- **Models Used**:
  - CNN Architectures: VGG16, ResNet18, DenseNet169
  - Transformer Architectures: Vision Transformer (ViT Base, ViT Large)
- **Plots**:
  - Training loss, training accuracy, and validation accuracy for each model.
- **Data**:
  - Dataset includes labeled images of melt pools (stable and unstable conditions). Due to privacy constraints, the dataset is not included in this repository. Instructions for data preparation are provided.
- **Results**:
  - Vision Transformers outperform CNNs in accuracy, with ViT Large achieving the highest validation accuracy of 98.70%.

## Getting Started

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/rojinta/MeltPoolAI.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Preparation
Since the dataset is not included, prepare it as follows:
- Organize your dataset into `train/` and `val/` subdirectories, each containing `stable/` and `unstable/` subfolders with corresponding images.
- Ensure the dataset follows this structure:
   ```
   data/
   ├── train/
   │   ├── stable/
   │   └── unstable/
   └── val/
       ├── stable/
       └── unstable/
   ```

### Running the Models
- Navigate to the `models/` folder and select the desired architecture.
- For example, to run the Vision Transformer Base model:
   ```bash
   python vit_base.py
   ```

## Results

|Model          | Validation Accuracy Before Noise (%) | Validation Accuracy After Noise (%) |
|---------------|--------------------------------------|-------------------------------------|
| VGG16         | 89.94                                | 96.22                              |
| ResNet18      | 82.14                                | 86.49                              |
| DenseNet169   | 84.09                                | 85.95                              |
| ViT Base      | 95.45                                | 96.22                              |
| ViT Large     | 98.70                                | 98.38                              |

- Vision Transformers demonstrated superior performance compared to CNNs due to their ability to capture global relationships in melt pool images.
- Adding Gaussian noise to the dataset improved the generalizability of nearly all models.
# OPS-SAT Project

## Dataset Information

- **Dataset Name:** The OPS-SAT case dataset
- **Dataset Variation Description:** Augmented Color Corrected Synthetic Variation

### Dataset Paths

- **Training/Validation Dataset Path:** ../Data/Variation_Synthetic_Generation_color_corrected_Augmentation/train/
- **Test Dataset Path:** ../Data/Variation_Synthetic_Generation_color_corrected_Augmentation/test/

## Model Configuration

- **Transfer Learning:** No
- **Transfer Learning Dataset:** landuse

### Model Parameters

- **Input Shape:** [200, 200, 3]
- **Number of Classes:** 8
- **Output Layer Activation:** Softmax
- **Model Optimizer:** Adam
- **Loss Function:** FocalLoss
- **Model Metrics:** [SparseCategoricalAccuracy]
- **Early Stopping:**
  - Monitor: val_sparse_categorical_accuracy
  - Patience: 6
- **Model Checkpoint:**
  - Monitor: val_sparse_categorical_accuracy
- **Cross Validation K-Fold:** 7
- **Number of Epochs:** 200
- **Batch Size:** 4
- **Focal Loss Parameters:**
  - Alpha: 0.2
  - Gamma: 2
- **Number of Freeze Layers:** 5

## Weights and Biases (wandb) Configuration

- **Project:** OPS-SAT-Thesis-Project
- **Config:**
  - Dropout: 0.5
  - Other parameters are consistent with the model configuration.

## Project Structure

```plaintext
- /OPS-SAT-Thesis-Project
  - /Data
    - /Variation_Synthetic_Generation_color_corrected_Augmentation
      - /train
      - /test
  - /src (or /code, /scripts, etc.)
    - Your source code files
  - README.md

  ## Supplementary Links

  - [ESA OPS-SAT Competition](https://kelvins.esa.int/opssat/home/)
  - [Dataset on Zenodo](https://zenodo.org/records/6524750)

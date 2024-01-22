# Few-shot Satellite Image Classification with OPS-SAT

Welcome to the Few-shot Satellite Image Classification repository using OPS-SAT! Follow the steps below to get started:

## Usage Guide

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/ShendoxParadox/Few-shot-satellite-image-classification-OPS-SAT.git
    ```

2. **Navigate to Repo Root Folder:**
    ```bash
    cd Few-shot-satellite-image-classification-OPS-SAT
    ```

3. **Build Docker Image:**
    ```bash
    docker build --no-cache -t ops_sat:latest .
    ```

4. **Run Docker Container:**
    ```bash
    docker run -it ops_sat
    ```

5. **Modify Configuration:**
    Edit the `config.json` file as needed:
    ```bash
    nano config.json
    ```

6. **Navigate to Source Folder:**
    ```bash
    cd src/
    ```

7. **Run OPS-SAT Development Script:**
    ```bash
    python OPS_SAT_Dev.py
    ```

8. **Choose W&B Option:**
    Follow the prompts to choose the WandB option during script execution.

9. **View Run Results:**
    Navigate to the WandB dashboard to observe the run results.

For any additional information or troubleshooting, refer to the documentation or contact the repository owner.


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

## Supplementary Links

- [ESA OPS-SAT Competition](https://kelvins.esa.int/opssat/home/)
- [Dataset on Zenodo](https://zenodo.org/records/6524750)

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



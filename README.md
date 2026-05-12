# Detection of Body-Focused Repetitive Behaviors using Deep Learning  
CSCI 6806 – Graduate Capstone Project Group 11  

**Team Members:**  
-Gan Song g.song@student.fdu.edu  
-Zixuan xu z.xu9@student.fdu.edu  
-Jinqiang Ding j.ding2@student.fdu.edu  
-Yuan Guo y.guo4@studnet.fdu.edu  
-Jia Min j.min@student.fdu.edu



GitHub Repository:  
`https://github.com/sourav-fdu-capstone/2026-winter-capstone-project-2026winter-capstone-group-11`

---

## 1. Overview

In this project, we reproduce the main models proposed in the paper:

Zhang, X., Ryoo, J., Mukherjee, S.,  
“Detection of Body-Focused Repetitive Behaviors using Deep Learning,” IEEE ComComAp 2025.

The first goal is to build a complete and reproducible pipeline that includes:

- Data ingestion
- Exploratory data analysis (EDA)
- Data cleaning and preprocessing
- Model implementation
- Evaluation using binary F1-score and macro-averaged F1-score

The dataset used in this project is the CMI Helios multi-modal sensor dataset. It includes three types of sensor data:

- IMU signals (motion data)
- Thermopile sensor readings
- TOF (Time-of-Flight) distance measurements

Each sensor captures different aspects of hand movement and proximity patterns related to body-focused repetitive behaviors (BFRBs).

---

In Milestone2, **Model 7 to Model 12** are our main contributions. These models explore NLP-inspired or sequential modeling ideas for sensor-based behavior recognition, including:
- Model 7: GRU
- Model 8: TOF-Only Transformer
- Model 9: LSTM with Attention
- Model 10: Temporal Convolution Network
- Model 11: TOF-Only Token LSTM
- Model 12: TOF-Only CNN Token LSTM

The goal of these models is to investigate whether NLP-style sequence modeling can better capture temporal patterns in sensor signals for BFRB detection. We evaluate all proposed models using the required metrics, including **binary F1-score** and **macro-averaged F1-score**, and compare their performance with the paper’s baseline methods such as **FFT-MLP**, **CNN-BiLSTM**, **Late Fusion**, **Intermediate Fusion**, and **FFT-Random Forest**.

This repository is organized to support **reproducibility and clear experiment tracking**. Each model is implemented in a separate Jupyter notebook, and the repository also includes notebooks for evaluation, result comparison, and plot generation. The overall objective is to provide a clear and reproducible study of how NLP-based modeling ideas can be applied to multimodal wearable sensor data for BFRB detection.

---


## 2. System Requirements

### Hardware

- CPU: x86_64 architecture
- RAM: ≥ 8 GB recommended
- GPU: Optional (recommended for CNN-BiLSTM training)

### Operating System

- Windows 11 (tested)
- Ubuntu 22.04 (expected compatible)
- macOS (expected compatible)

### Software

- Python 3.10.x
- Anaconda (recommended)
- Git
- Jupyter Notebook

### Python Dependencies

The project was tested with the following package versions:

```
python==3.10.12
numpy==1.24.4
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
scipy==1.11.2
tqdm==4.66.1
torch==2.1.0
torchvision==0.16.0
notebook==7.0.6
ipykernel==6.25.2
```

These packages can be installed using:

```
pip install -r requirements.txt
```
### Data Files  
- **Dataset**: `kaggle competitions download -c cmi-detect-behavior-with-sensor-data`
- **Size**: ~1.12 G
- **Location**: `data/raw/201910/train.csv`

## 3. Setup Instructions

Estimated total setup time: 15–25 minutes.

---

### Step 1 – Clone Repository

Clone the repository from GitHub:

```
git clone https://github.com/sourav-fdu-capstone/2026-winter-capstone-project-2026winter-capstone-group-11
cd 2026-winter-capstone-project-2026winter-capstone-group-11
```

Time estimate: ~1 minute

---

### Step 2 – Create Python Environment (Recommended: Conda)

Create a clean environment to avoid dependency conflicts:

```
conda create -n Group11 python=3.10
conda activate Group11
```

Time estimate: ~3 minutes

---

### Step 3 – Install Dependencies

Install all required packages:

```
pip install -r requirements.txt
```

Time estimate: ~5–10 minutes

---

### Step 4 – Download Dataset

Download the CMI Helios dataset from Kaggle.  
`https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/data`  
or command `kaggle competitions download -c cmi-detect-behavior-with-sensor-data`  
Place the train.csv file inside:

```
data/raw/
```

Example directory structure:

```
project-root/
│
├── data/
│   ├── raw/
│   │   └── train.csv
```

Time estimate: depends on internet speed (~5–10 minutes)

---

### Step 5 – Launch Jupyter Notebook

Start Jupyter:

```
jupyter notebook
```

Open the `notebooks/` directory and run notebooks in order.

---

### Step 6 – Verify Environment

Run:

```
bash check_environment.sh
```

Expected output:

```
✓ Python 3.10.x
✓ numpy installed
✓ pandas installed
✓ matplotlib installed
✓ scikit-learn installed
✓ torch installed
✓ data/raw directory exists

Environment check PASSED
```

Time estimate: ~1 minute  
Total setup time: ~25–30 minutes


## 4. Repository Structure

```
.
├── .github/
│ └── CODEOWNERS
│
├── A1 Report/
│ └── README.md
│
├── artifacts/
│ ├── fft_mlp_all/            
│ ├── cnn_bilstm_tof_seq/
| ├── Model9_LSTM_Attention
│ └── Model5_Training/
│
├── data/
│ ├── raw/ # Raw Kaggle data (not included)        ## Raw Data should be download and put train.csv here
│ ├── processed/                                   ## Cleaned & normalized parquet files
│ │ ├── train_normalized.parquet                                   
│ │ ├── test_normalized.parquet
│ │ └── Processed_FFT/
│ └── processed_data_tof/                           ## TOF-specific numpy datasets
│ └──Processed_TOF_Transformer/
|
├── notebooks/
│ ├── 01_preprocessing.ipynb            ## For EDA process, you should run 01_preprocessing.ipynb first and 02_eda.ipynb next
│ ├── 02_eda.ipynb
│ ├── STEP_0 Data Optimization and Organization.ipynb    ## Step 0,1,2 are processing step for next modeling
│ ├── STEP_1 Data Normalization.ipynb
│ ├── STEP_2 Data_FFT.ipynb
│ │
│ ├── modeling/
│ │ ├── Model_1_FFT_MLP (All Inputs).ipynb            ##  All model files are here
│ │ ├── Model_2_FFT_MLP.ipynb
│ │ ├── Model_3_CNN_BiLSTM.ipynb
│ │ ├── Model_4_Late_Fusion_Ensemble.ipynb
│ │ ├── Model 5 (Intermediate Fusion).ipynb
| | ├── Model 6 FFT–Random Forest.ipynb
| | ├── Model 7 Model_7_GRU.ipynb
| | ├── Model 8 Model_8_TOF_Only_Transformer_Model.ipynb
| | ├── Model 9 Model_9_LSTM_Attention.ipynb
| | ├── Model 10 Model_10_Temporal_Conv_Network.ipynb
| | ├── Model 11 Model_11_TOF_Only_Token_LSTM.ipynb
│ │ └── Model 12 Model_12_TOF_Only_CNN_Token_LSTM.ipynb
│ │
| |
│ └── evaluation/
│ └── Evaluations.ipynb                  ## After all processes finished, this ipynb is to Evaluate all output saved in results
│
├── plots/                                ## All plots are saved here
│ ├── fft_mlp_all/
│ ├── model3_cnn/
│ ├── evaluation_plots/
│ ├── bfrb_vs_nonbfrb_modality_plots/
│ ├── model2_training_curves/
│ ├── Model 5 (Intermediate Fusion)/
│ |── Model 6 FFT–Random Forest/
| ├── Model4
| ├── model7_gru
| ├── Model_8_tof_transformer
| ├── Model9_LSTM_Attention
| ├── Model_10_TCN
| ├── model_11_training_curves
| ├── model_12_training_curves
| └── eda
│
├── results/                          ## Every model has an output and wait for evaluate
│ ├── Model1_Training/
│ ├── Model2_Training/
│ ├── Model3_Training/
│ ├── Model4_Training/
│ ├── Model5_Training/
│ ├── Model6_Training/
| ├── Model7_Training/
| ├── Model8_Training/
| ├── Model9_Training/
| ├── Model10_Training/
| ├── Model11_Training/
| ├── Model12_Training/
│ ├── models/
│ └── preds/
│
├── check_environment.sh                                        
├── requirements.txt
└── README.md

```






## 5. Reproducing Results

All experiments are organized as a sequence of Jupyter notebooks.  
Each notebook saves its output to disk so that the next notebook can read it.  
This ensures the entire pipeline can be reproduced from a fresh clone.

Please run the notebooks in the following order.

---

### Step 1 – Data Preprocessing

Run:

- `notebooks/STEP_0 Data Optimization and Organization.ipynb`

This notebook:
- Removes NULL values
- Filters short sequences

```
Saved in:
  - results/preds/id2label.json
  - results/preds/label2id.json
  - results/preds/split_test_seq_ids.npy
  - results/preds/split_train_seq_ids.npy
  - data/processed/final_processed_data.parquet
```

---

Time estimate: ~1–2 minutes  

### Step 2 – Exploratory Data Analysis (EDA)

Run:

- `notebooks/01_preprocessing.ipynb `
- `notebooks/02_eda.ipynb`

This notebook:
* Loads and profiles data: Imports the raw training dataset and identifies the total number of rows and columns.
* Analyzes missing values:
    * Computes the percentage of missing values for every feature.
    * Calculates the "Row Missing Ratio" to determine the percentage of rows containing at least one null value.
**Benchmarks sensor reliability**:
    * Groups features by sensor type: **IMU** (acc/rot), **Thermopile** (thm), and **TOF** (tof).
    * Visualizes the average missing value percentage per sensor group.
* **Evaluates sequence dynamics**:
    * Computes sequence lengths based on `sequence_id`.
    * Generates descriptive statistics and histograms for sequence length distributions.
* **Validates preprocessing**: Compares sequence length distributions between **Raw** and **Cleaned** datasets to assess the impact of data cleaning.

**Outputs:**  
`saved in plots and data/processed`
  - Raw vs Cleaned Distribution
  - Sequence Length Distribution
  - Missing Values by Sensor Group
```
data/processed/train_cleaned.csv
plots/eda/missing_values_by_sensor.png
plots/eda/missing_values_top_20.png
plots/eda/sequence_length_comparison.png
plots/eda/sequence_length_dist_raw.png
```

Time estimate: ~1–2 minutes  

---

### Step 3 – Data Normalization

Run:

- `notebooks/STEP_1 Data Normalization.ipynb`

This notebook:
- split the cleaned data  
- Normalization data

Outputs:
- `train_subject_split.parquet`
- `test_subject_split.parquet`
- `train_normalized.parquet`
- `test_normalized.parquet`
- `Normaliaztion_accx_thm1.png`

Saved in:

```
data\processed\test_subject_split.parquet
data\processed\train_subject_split.parquet
data\processed\train_normalized.parquet
data\processed\test_normalized.parquet
plots\bfrb_vs_nonbfrb_modality_plots\Normaliaztion_accx_thm1.png
```

Time estimate: ~1–2 minutes  

---

### Step 4 – Data FFT + FFT Feature Visualization

Run:

- `notebooks/STEP_2_Data_FFT.ipynb`

This notebook:

- Loads normalized train/test datasets
- Applies FFT transformation to each sequence
- Pads or truncates each sequence to length 64
- Keeps the first 32 frequency components
- Flattens FFT features into 1D vectors
- Generates both 18-class labels and binary labels
- Saves processed FFT features for model training
- Generates modality comparison plots (IMU vs THM vs TOF)

Outputs:

- `data_train.npy`
- `label_train_18.npy`
- `label_train_binary.npy`
- `data_test.npy`
- `label_test_18.npy`
- `label_test_binary.npy`
- `gesture_classes.npy`
- `nonbfrb_seq_XXX_imu_thm_tof.png`
- `bfrb_seq_XXX_imu_thm_tof.png`

Saved in:

```
data/processed/Processed_FFT/data_train.npy
data/processed/Processed_FFT/label_train_18.npy
data/processed/Processed_FFT/label_train_binary.npy
data/processed/Processed_FFT/data_test.npy
data/processed/Processed_FFT/label_test_18.npy
data/processed/Processed_FFT/label_test_binary.npy
data/processed/Processed_FFT/gesture_classes.npy

plots/bfrb_vs_nonbfrb_modality_plots/nonbfrb_seq_XXX_imu_thm_tof.png
plots/bfrb_vs_nonbfrb_modality_plots/bfrb_seq_XXX_imu_thm_tof.png
```

Time estimate: ~2–5 minutes  

---

### Step 5 – Model 1 FFT-MLP_All_INUPT

Run:

- `notebooks/Model_1_FFT_MLP (All Inputs).ipynb`

This notebook:
- Builds an MLP model (3 hidden layers: 128 → 64 → 64) with an 9-class output layer
- Trains the model using:
  - Optimizer: Adam (lr = 0.001)
  - Epochs: 20
  - Batch size: 256
  - Random seed: 42
- Evaluates performance on the test set each epoch:
  - **Binary F1-score** (BFRB vs non-BFRB)
  - **Macro F1-score (9-class)** (8 BFRBs + 1 non-BFRB group)
- Prints per-epoch training loss, test loss, Binary F1, and Macro F1

Outputs:

- Model checkpoint
- Training and testing loss history
- Binary F1 and Macro F1 scores
- Prediction results (logits, probabilities, true and predicted labels)
- Training & Test Loss Curve
- Binary F1 Curve
- Macro F1 (9-class) Curve


Saved in:

```
artifacts/fft_mlp_all/checkpoint.pt
artifacts/fft_mlp_all/metrics.json
artifacts/fft_mlp_all/model_weights.pt
artifacts/fft_mlp_all/test_f1_bin.npy
artifacts/fft_mlp_all/test_f1_macro9.npy
artifacts/fft_mlp_all/test_losses.npy
artifacts/fft_mlp_all/train_losses.npy
artifacts/fft_mlp_all/y_logits.npy
artifacts/fft_mlp_all/y_pred_18.npy
artifacts/fft_mlp_all/y_prob.npy
artifacts/fft_mlp_all/y_true_18.npy
artifacts/fft_mlp_all/binary_f1_curve.png
artifacts/fft_mlp_all/loss_curve.png
artifacts/fft_mlp_all/macro_f1_9class_curve.png
```

Time estimate: ~2–5 minutes

---

### Step 6 – Model 2 FFT_MLP (IMU and THM Only)

Run:

- `notebooks/modeling/Model_2_FFT_MLP.ipynb`

This notebook:
Extracts FFT features from time-domain IMU and THM sensors only (excluding TOF)
- Pads/truncates sequences to a fixed length (L=59) and standardizes the frequency-domain features
- Builds an MLP model (3 hidden layers: 128 → 64 → 64) with an 9-class output layer
- Trains the model using:
  - Optimizer: Adam (lr = 0.001)
  - Epochs: 20
  - Batch size: 256
  - Random seed: 42
- Evaluates performance on a global test set (1518 sequences) each epoch:
  - Binary F1-score (BFRB vs non-BFRB)
  - Macro F1-score (9-class)
  - Final Epoch metrics
- Prints per-epoch training loss, test loss, Binary F1, and Macro F1

Outputs: 
- Model checkpoint
- Prediction results (test logits, true labels, sequence IDs, and class mappings)
- Training summary JSON (contains final accuracy and F1 metrics)
- Training and Test Loss Curve
- Binary F1 Curve
- Macro F1 (9-class) Curve

```
results/models/fft_mlp_imu_thm_only.pth
results/preds/fft_mlp_imu_thm_only_classes.npy
results/preds/fft_mlp_imu_thm_only_test_logits.npy
results/preds/fft_mlp_imu_thm_only_test_seq_ids.npy
results/preds/fft_mlp_imu_thm_only_test_y.npy
results/Model2_Training/model2_training_summary.json
plots/model2_training_curves/binary_f1_curve.png
plots/model2_training_curves/loss_curve.png
plots/model2_training_curves/macro_f1_curve.png
```

Time estimate: ~2–5 minutes

---

### Step 7 – Model 3 CNN-BiLSTM (TOF Only)

Run:

- `notebooks/modeling/Model_3_CNN_BiLSTM.ipynb`

This notebook:
- Extracts spatial and temporal features from Time-of-Flight (TOF) sensors only
- Reshapes 64 TOF features into an 8x8 spatial grid per time step and pads/truncates sequences to a fixed length (L = 59)
- Builds a CNN-BiLSTM model:
  - CNN extracts spatial features per frame (2 Conv2d layers with MaxPool)
  - BiLSTM captures temporal dynamics (2 bidirectional layers, hidden size = 64)
  - MLP classifier maps the sequence representation to an 9-class output layer
- Trains the model using:
  - Optimizer: Adam (lr = 0.001)
  - Epochs: 20
  - Batch size: 256
  - Random seed: 42
- Evaluates performance on a global test set (1518 sequences) each epoch:
  - Binary F1-score (BFRB vs non-BFRB)
  - Macro F1-score (9-class)
  - Final Epoch metrics
- Prints per-epoch training loss, test loss, Binary F1, and Macro F1

Outputs:
- Model checkpoint
- Prediction results (test logits, true labels, sequence IDs, and class mappings)
- Training summary JSON (contains final accuracy and F1 metrics)
- Training and Test Loss Curve
- Binary F1 Curve
- Macro F1 (9-class) Curve

```
results/preds/cnn_bilstm_tof_only_classes.npy
results/preds/cnn_bilstm_tof_only_test_logits.npy
results/preds/cnn_bilstm_tof_only_test_seq_ids.npy
results/preds/cnn_bilstm_tof_only_test_y.npy
results/Model3_Training/model3_training_summary.json
plots/model3_cnn/binary_f1_curve.png
plots/model3_cnn/loss_curve.png
plots/model3_cnn/macro_f1_curve.png
```

Time estimate: ~3–8 minutes

---
### Step 8 – Model 4 Late Fusion Ensemble

Run:

- `notebooks/modeling/Model_4_Late_Fusion_Ensemble.ipynb`

This notebook:

- Loads the npy files of test logits, ground-truth labels, and sequence IDs from Model 2 (IMU + Thermopile only), and Model 3 (CNN-BiLSTM TOF)
- Loads the json file id2label.json that is produced by the file Data Preprocessing
- Perform logits-level late fusion as described in the paper. The raw (pre-softmax) logits from Model 2 and Model 3 are combined using a weighted sum (0.3 / 0.7)
- A numerically stable softmax is then applied to obtain final class probabilities and 18-class predictions
- Derive the set of BFRB classes from the label definitions (id2label.json)
- Evaluates the final ensemble performance on the global test set (1518 sequences):
  - Binary F1-score (BFRB vs non-BFRB)
  - Macro F1-score (9-class)
  - Final Ensemble metrics
- Prints the detailed 9-class classification report and binary classification summary

Outputs:

- Ensemble metrics summary JSON (contains final accuracy and F1 metrics)
- Macro F1(9-class) plot
- Binary F1 (BFRB vs non-BFRB) plot
- Fused raw logits

Saved in:

```
results/preds/late_fusion_test_logits.npy
results/preds/late_fusion_test_pred18.npy
results/Model4_Training/model4_training_summary.json
plots/model4/model4_binary_f1.png
plots/model4/model4_macro_f1.png
```

Time estimate: ~2–5 minutes

---
### Step 9 – Model 5 (Intermediate Fusion)
Run:

- `notebooks/modeling/Model 5 (Intermediate Fusion).ipynb`

This notebook implements the Intermediate Fusion ensemble described in the paper, combining:
- FFT-MLP branch (IMU + Thermopile only)
- CNN-BiLSTM branch (TOF only)

Key pipeline steps:
- Loads the existing normalized split directly from STEP_1 outputs:
  - data/processed/train_normalized.parquet
  - data/processed/test_normalized.parquet

- Builds sequence-level FFT features for IMU + THM:
  - K_FREQ = 32 (keeps first 32 frequency components)
  - Uses rfft magnitude and flattens into 1D feature vector per sequence

- Packs TOF data into fixed-length tensors:
  - MAX_LEN = 48 timesteps (pad/truncate)
  - reshapes to (L, 5, 8, 8) per sequence (5 TOF sensors x 8x8 each)

- Training setup:
  - Loss: CrossEntropy
  - Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
  - Epochs: 20
  - Batch size: 64
  - Seed: 42

- Evaluation each epoch:
  - Accuracy
  - Macro F1 (9-class)
  - Binary F1 (BFRB vs non-BFRB; BFRB labels defined in notebook)
- Best model selection: chooses the checkpoint with highest Macro F1

Saved in:

```
  - artifacts/Model5_Training/best_macro_*.pt
  - results/Model5_Training/model5_training_summary.json
  - results/Model5_Training/model5_training_history.csv
  - plots/Model 5 (Intermediate Fusion)/model5_train_loss.png
  - plots/Model 5 (Intermediate Fusion)/model5_accuracy.png
  - plots/Model 5 (Intermediate Fusion)/model5_macro_f1.png
  - plots/Model 5 (Intermediate Fusion)/model5_binary_f1.png

```

Time estimate: ~2–5 minutes

---
### Step 10 – Model 6 Model 6 FFT–Random Forest
Run:
  - `notebooks/modeling/Model 6 FFT–Random Forest.ipynb`

- This notebook implements the FFT + Random Forest classical baseline (paper Model 6).

- Key pipeline steps:
- Loads:
  - data/processed/final_processed_data.parquet

- Preprocessing (same idea as earlier steps):
  - Drops rows with NULL values across sensor columns
  - emoves very short sequences (length < mean - std)

- Train/test split:
  - Random 80/20 split by sequence_id with seed = 42

- Builds sequence-level FFT features using all sensors (IMU + THM + TOF):
  - K_FREQ = 32
  - Applies rfft magnitude per sequence, pads/truncates to 32 bins
  - Flattens to 1D feature vector per sequence

- Trains Random Forest:
  - n_estimators = 100
  - max_depth = None
  - random_state = 42
  - Majority voting is the default RF decision rule

- Evaluation:
  - Accuracy (multi-class)
  - Macro F1 (multi-class)
  - Binary F1 (BFRB vs non-BFRB)

- Outputs:
  - Metrics printed in notebook
  - Bar chart for metrics
  - Multi-class confusion matrix

Saved in:
```
  - plots/Model 6 FFT-Random Forest/model6_metrics.png
  - results/Model6_Training/model6_training_summary.json
  - plots/Model 6 FFT-Random Forest/model6_confusion_matrix.png
```

Time estimate: ~2–5 minutes


---

## 6. Experimental Results Based on NLP Models

---

### Model 7 – GRU-based Multimodal Sequence Model

Run:
- notebooks/modeling/Model_7_GRU.ipynb
  
This model implements a GRU-based architecture for multivariate time-series classification using IMU, thermopile, and TOF data.

#### Training Setup
- Optimizer: Adam  
- Learning rate: 0.001  
- Batch size: 128  
- Epochs: 20  

#### Performance
- Binary F1 improved from ~0.7x to ~0.92  
- Based on 10+ experiments, 7 stable runs selected  

#### Outputs
- results/Model7_Training/results.json
- plots/Model 7/

---

Model 8 TOF-Only Time-Series-Transformer Model
---

### Model 8 TOF-Only Time-Series-Transformer Model

#### Step 1 – TOF-only sequence preprocessing
Run:
  - `notebooks/modeling/TOF-only sequence preprocessing.ipynb`

- This notebook preprocesses TOF-only sensor data for the Transformer-based sequence model.

- Key pipeline steps:
  - Loads:
    - `data/processed/train_normalized.parquet`
    - `data/processed/test_normalized.parquet`

- Preprocessing:
  - Selects TOF columns only
  - Groups rows by `sequence_id` so each gesture sequence becomes one sample
  - Pads or truncates each sequence to a fixed length:
    - `TARGET_LEN = 128`
  - Builds:
    - TOF feature arrays
    - Padding masks
    - 18-class gesture labels
    - Binary labels (BFRB vs non-BFRB)
    - Original effective sequence lengths

- Outputs:
  - Processed NumPy arrays for Transformer training

Saved in:
```
-data/processed/Processed_TOF_Transformer/data_train_tof.npy
-data/processed/Processed_TOF_Transformer/mask_train_tof.npy
-data/processed/Processed_TOF_Transformer/label_train_18.npy
-data/processed/Processed_TOF_Transformer/label_train_binary.npy
-data/processed/Processed_TOF_Transformer/data_test_tof.npy
-data/processed/Processed_TOF_Transformer/mask_test_tof.npy
-data/processed/Processed_TOF_Transformer/label_test_18.npy
-data/processed/Processed_TOF_Transformer/label_test_binary.npy

```

Time estimate: ~2–5 minutes

---

#### Step 2 - TOF-Only Transformer Model
Run:
  - `notebooks/modeling/Model_8_TOF_Only_Transformer_Model.ipynb`

- This notebook implements a Transformer-based TOF-only sequence model for BFRB gesture classification.

- Key pipeline steps:
  - Loads:
    - `data/processed/Processed_TOF_Transformer/data_train_tof.npy`
    - `data/processed/Processed_TOF_Transformer/mask_train_tof.npy`
    - `data/processed/Processed_TOF_Transformer/label_train_18.npy`
    - `data/processed/Processed_TOF_Transformer/label_train_binary.npy`
    - `data/processed/Processed_TOF_Transformer/data_test_tof.npy`
    - `data/processed/Processed_TOF_Transformer/mask_test_tof.npy`
    - `data/processed/Processed_TOF_Transformer/label_test_18.npy`
    - `data/processed/Processed_TOF_Transformer/label_test_binary.npy`

- Model:
  - Input dimension = 320
  - Transformer hidden size (`d_model`) = 128
  - Number of attention heads = 4
  - Number of Transformer encoder layers = 2
  - Feedforward dimension = 256
  - Dropout = 0.1
  - Output classes = 18

- Training:
  - Batch size = 128
  - Optimizer = Adam
  - Learning rate = 0.0008
  - Epochs = 20
  - Random seed = 42

- Evaluation:
  - Accuracy (18-class)：0.481
  - Macro F1 ：0.486
  - Binary F1 (BFRB vs non-BFRB) ：0.938

- Outputs:
  - Training/test loss curve
  - Binary F1 curve
  - Macro F1 curve
  - Summary JSON with best epoch and final metrics

Saved in:
```
-plots/tof_transformer/loss_curve.png
-plots/tof_transformer/binary_f1_curve.png
-plots/tof_transformer/macro_f1_curve.png
-artifacts/tof_transformer/
-results/Model7_Training/model7_training_summary.json

```

Time estimate: ~2–5 minutes

---

### Model 9
Run:
  - `notebooks/modeling/Model_9_LSTM_Attention.ipynb`

- This notebook implements LSTM Attention model: a multimodal BFRB sequence detection model inspired by natural language processing.

- Key pipeline steps:
  - Loads:
    - `data/processed/train_clean.parquet`
    - `data/processed/train_sequence_ids.csv`
    - `data/processed/Processed_TOF_Transformer/label_train_18.npy`

- Sequence construction:
  - Groups data by sequence_id
  - Builds variable-length sequences using all sensors:
  - IMU + THM + TOF
  - Each sample has shape: (T, num_features)
  - Applies feature standardization based on training data only

- Handles variable-length sequences:
  - Uses padding to align sequences within each batch
  - Applies masking to ignore padded time steps
  - Uses pack_padded_sequence for efficient LSTM computation

- Model:
  - LSTM encoder:
    - hidden_dim = 128
    - num_layers = 1
  - Attention mechanism:
    - Computes importance weights across time steps
    - Aggregates sequence features using weighted sum
  - Classification:
    - Fully connected layer + softmax

- Training:
  - Optimizer: AdamW
  - Learning rate: 1e-3
  - Batch size: 64
  - Epochs: 20

- Evaluation:
  - Accuracy (18-class)：0.6034
  - Macro F1 ：0.6009
  - Binary F1 (BFRB vs non-BFRB) ：0.9725

- Outputs:
  - Training/test loss curve
  - Binary F1 curve
  - Macro F1 curve
  - Summary JSON with best epoch and final metrics

Saved in:
```
- artifacts/Model9_LSTM_Attention/model9_lstm_attention_best.pt
- results/Model9_LSTM_Attention/model9_training_summary.json
- results/Model9_LSTM_Attention/model9_training_history.csv
- plots/Model9_LSTM_Attention/model9_lstm_attention_loss.png
- plots/Model9_LSTM_Attention/model9_lstm_attention_metrics.png
- plots/Model9_LSTM_Attention/model9_lstm_attention_confusion_matrix.png

```

Time estimate: ~15–30 minutes

---


### Model 10: Temporal Convolutional Network (TCN)

This section describes how to reproduce the results of Model 10, a Temporal Convolutional Network (TCN) for BFRB detection using IMU and THM features.

### File Location

notebooks/modeling/Model_10_Temporal_Conv_Network.ipynb

### Output Locations

plots/Model_10_TCN/  
results/Model10_Training/model10_tcn_results.json

---

### Reproducing Model 10 Results

### Step 1. Clone the repository

git clone https://github.com/sourav-fdu-capstone/2026-winter-capstone-project-2026winter-capstone-group-11.git

---

### Step 2. Create and activate the Python environment

conda create -n capstone python=3.10 -y  
conda activate capstone

---

### Step 3. Install dependencies

pip install -r requirements.txt  

If needed, install manually:

pip install numpy pandas matplotlib scikit-learn torch pyarrow jupyter

---

### Step 4. Launch Jupyter Notebook

jupyter notebook

---

### Step 5. Open the notebook

Open the following file:

notebooks/modeling/Model_10_Temporal_Conv_Network.ipynb

---

### Step 6. Run all cells in order

Run all cells from top to bottom. The notebook performs the following steps:

- Loads normalized training data from data/processed/train_normalized.parquet  
- Selects IMU and THM features  
- Groups data by sequence_id to construct fixed-length sequences  
- Encodes gesture labels for 18-class classification  
- Trains the TCN model  
- Evaluates validation Macro F1 and Binary F1  
- Saves plots and JSON results  

---

### Step 7. Check generated plots

All plots are saved to:

plots/Model_10_TCN/

Generated plots include:

- Loss curve  
- Validation Macro F1 curve  
- Final metrics plot  
- Confusion matrix  

---

### Step 8. Check saved results

Results are saved to:

results/Model10_Training/model10_tcn_results.json

The JSON file includes:

- model name  
- best epoch  
- best Macro F1  
- best Binary F1  
- training loss history  
- validation loss history  
- Macro F1 progression across epochs  

---

### Step 9. Final validation performance

Best Epoch: 14  
Validation Macro F1: 0.4304  
Validation Binary F1: 0.9568  

---

### Step 10. Model comparison

The outputs from Model 10 can be used to compare against other models such as GRU and Transformer using:

- Macro F1  
- Binary F1  

---

## Notes

- Run the notebook in order without skipping cells  
- Do not commit large data files such as .parquet, .csv, or .npy  
- Ensure correct relative paths when running the notebook  
- Re-running the notebook will overwrite existing plots and JSON results  

---
### Model 11
Run:
  - `notebooks/modeling/Model_11_TOF_Only_Token_LSTM.ipynb`

- This notebook implements a Token-based LSTM model: an NLP-inspired BFRB sequence detection model using TOF-only sensor data.
- Instead of using raw sensor values directly, this model discretizes TOF features into tokens, enabling sequence modeling similar to natural language processing.

- Key pipeline steps:
  - Loads:
    - `data/processed/clean_chunks22/*.parquet`
    - `results/preds/split_train_seq_ids.npy`
    - `results/preds/split_test_seq_ids.npy`
    - `data/processed_data_tof/gesture_classes_raw.npy` 

- Sequence construction:
  - Groups data by sequence_id
  - Uses canonical split (6070 train / 1518 test)
  - Builds variable-length sequences using:TOF only (320 features per timestep)

- Tokenization (core idea):
  - For each time step, extracts statistical features: mean / std / min / max
  - Discretizes each feature into bins (fit on training data only)
  - Combines them into a joint token ID: (mean_bin, std_bin, min_bin, max_bin) → single token
  - Final sequence becomes: token sequence of shape (T,)

- Handling variable-length sequences:
  - Applies padding / truncation: MAX_LEN = 100
  - Uses padding token (PAD_TOKEN_ID = 0) 
  - Keeps original sequence lengths for masking

- Model:
  - Embedding layer:
    - hidden_dim = 128
    - num_layers = 1
    - Uses pack_padded_sequence for efficiency
  - Sequence representation:
    - Uses last hidden state
  - Classification:
    - Fully connected layer → logits (18 classes)

- Training:
  - Optimizer: Adam
  - Learning rate: 1e-3
  - Batch size: 64
  - Epochs: 30
  - Uses:
    - Cross-entropy loss   
    - Validation-based evaluation

- Evaluation:
  - Macro F1 (9-class): ~0.28–0.31
  - Binary F1 (threshold tuned): ~0.84–0.85
  - Binary F1 (argmax): slightly lower
  - Binary setup:
    - BFRB = positive
    - non-BFRB = negative 

- Outputs:
  - Training / validation / test loss curves
  - Binary F1 curve
  - Macro F1 (9-class) curves
  - Prediction logits / labels / sequence IDs
  - Summary JSON with final metrics

Saved in:
```
- results/Model11_Training/Model_11_TOF_Only_Token_LSTM_test_logits.npy
- results/Model11_Training/Model_11_TOF_Only_Token_LSTM_test_seqids.npy
- results/Model11_Training/Model_11_TOF_Only_Token_LSTM_test_y.npy
- results/Model11_Training/model11_training_summary.json
- plots/model_11_training_curves/binary_f1_curve.png
- plots/model_11_training_curves/macro_f1_curve.png
- plots/model_11_training_curves/loss_curve.png
```

Time estimate: ~15–30 minutes

---

### Model 12
Run:
  - `notebooks/modeling/Model_12_TOF_Only_CNN_Token_LSTM.ipynb`

- This notebook implements a CNN + Token-like embedding + BiLSTM model: an NLP-inspired BFRB sequence detection model using TOF-only sensor data.
- Instead of using raw sensor values directly for classification, this model first reshapes each TOF frame into spatial grids, applies a CNN encoder to extract frame-level features, and     then models temporal dependencies across frames using a BiLSTM.
- Model 12 is not a traditonally text-based NLP model, it adopts several NLP-style sequence modeling ideas: token-like frame embeddings, variable-length sequence padding, packed sequence    processing, and BiLSTM contextual modeling.

- Key pipeline steps:
  - Loads:
    - `data/processed/clean_chunks22/*.parquet`
    - `results/preds/split_train_seq_ids.npy`
    - `results/preds/split_test_seq_ids.npy`
    - `data/processed_data_tof/gesture_classes_raw.npy` 

- Sequence construction:
  - Groups data by sequence_id
  - Uses canonical split (6070 train / 1518 test)
  - Builds variable-length sequences using:TOF only (320 features per timestep)

- TOF frame reshaping:
  - For each time step, reshapes TOF input: 320 → (5, 8, 8)
  - This converts each timestep into a 5-channel spatial frame, where each TOF sensor contributes an 8 × 8 map.

- Handling variable-length sequences:
  - Uses train / validation split on training data
  - Applies padding / truncation: MAX_LEN = 100
  - Uses dynamic batch padding in the collate function
  - Keeps original sequence lengths for masking
  - Uses pack_padded_sequence for efficient sequence modeling

- Model:
  - CNN encoder:
    - Applies 2D CNN to each TOF frame:
      - Conv2d(5, 32, kernel_size=3, padding=1)
      - BatchNorm2d
      - ReLU
      - MaxPool2d
      - Conv2d(32, 64, kernel_size=3, padding=1)
      - BatchNorm2d
      - ReLU
      - MaxPool2d

  - Projects each frame into a compact embedding:
    - Flatten
    - Linear → token_dim = 128
    - ReLU
    - Dropout
      
  - Sequence encoder:
    - BiLSTM:
      - hidden_dim = 128
      - num_layers = 1
      - bidirectional = True

  - Sequence representation: Uses the final hidden state from both directions
 
  - Classification: Fully connected layers → logits (18 classes)
    
- Training:
  - Optimizer: AdamW
  - Learning rate: 5e-4
  - Weight decay: 1e-4
  - Batch size: 32
  - Epochs: 30
  - Early stopping patience: 6
  - Gradient clipping: 1.0
  - Uses:
    - Cross-entropy loss   
    - Validation-based threshold tuning  
    - Early stopping based on validation binary F1
      
- Evaluation:
  - Macro F1 (9-class): 0.6070
  - Binary F1 (threshold tuned): 0.9515
  - Binary setup:
    - BFRB = positive
    - non-BFRB = negative 

- Outputs:
  - Training / validation / test loss curves
  - Binary F1 curve
  - Macro F1 (9-class) curves
  - Prediction logits / labels / sequence IDs
  - Summary JSON with final metrics

Saved in:
```
- results/Model12_Training/Model_12_TOF_Only_CNN_Token_LSTM_test_logits.npy
- results/Model12_Training/Model_12_TOF_Only_CNN_Token_LSTM_test_seqids.npy
- results/Model12_Training/Model_12_TOF_Only_CNN_Token_LSTM_test_y.npy
- results/Model12_Training/model12_training_summary.json
- plots/model_12_training_curves/loss_curve.png
- plots/model_12_training_curves/binary_f1_curve.png
- plots/model_12_training_curves/macro_f1_curve.png
```

Time estimate: ~15–30 minutes

---


## 7. Final Evaluation

Run:
  - `notebooks/evaluation/Evaluations.ipynb`

This section evaluates the FFT + Random Forest baseline (paper Model 6) on the held-out test set, using the same task definitions and metrics reported in the paper.

In Milestone 2, we calculate the Binary F1 and Marco for NLP models which is Model 7-12, then we compare the results with baseline which is the models in the paper.

- Metrics Computed
  - Accuracy
  - Macro F1 (multi-class)
  - Binary F1 (BFRB vs non-BFRB)

- Visualizations Produced
  - Bar chart of metrics (`Accuracy`, `Macro F1`, `Binary F1`)
  - 4 figtures that produced in original paper
  - 2 figtures that show the difference Binary and Marco F1 compare with the paper

Saved in:
```
  - plots/evaluation_plots/binaryF1_comparison.png
  - plots/evaluation_plots/macroF1_comparison.png
  - plots/evaluation_plots/accuracy_comparison.png
  - plots/evaluation_plots/binaryF1_all_sensor_models.png"
  - plots/evaluation_plots/macroF1_all_sensor_models.png
  - plots/evaluation_plots/binaryF1_fft_subsets.png
  - plots/evaluation_plots/binaryF1_constituents_vs_ensembles.png
  - plots/evaluation_plots/model7_to_12_vs_baselines_binary_f1.png
  - plots/evaluation_plots/model7_to_12_vs_baselines_Marco_f1.png
```

Time estimate: ~2–5 minutes


---
### Expected Outputs

After completing all steps:

- The `plots/` directory should contain all of our output plots.
- The `data/processed/` directory should contain intermediate files (.npy or .parquet).
- The `results/` directory should contain all of our output scores.
- Final evaluation metrics (Binary F1 and Macro F1) should be displayed in the last notebook and saved in `plots/evaluation_plots`

If all notebooks run without errors, the results are successfully reproduced.

---

## Limitations

1. **Fixed Thresholding**  
   For simplicity in our code, we use a fixed threshold of 0.5 to convert predicted probabilities into binary labels. In a real-world clinical setting, this threshold might need to be adjusted to be more or less sensitive depending on the specific patient's needs.

2. **Single-Device Dependency**  
  The models were trained specifically on data from the Helios wristband. Because the system relies on specific multi-modal inputs—like the 5 Time-of-Flight (TOF) sensors and thermopiles—it might not work on standard smartwatches that only have an IMU

3. **Fixed train-test split**  
  We use a fixed 80:20 train-test split for all experiments. Cross-validation or leave-one-subject-out validation is not implemented in this version.

4. **Environmental Temperature Noise**  
   The thermopile sensors (TMP) measure skin temperature to help identify body parts. However, our model does not account for changes in room temperature or weather, which could confuse the sensors since they rely on detecting subtle heat differences.
   
5. **Fixed FFT Windowing**  
   Our preprocessing maps sequences to the frequency domain using the Fast Fourier Transform (FFT). Because we use a fixed approach, we might lose the specific "timing" of when a gesture starts or ends within a data clip, focusing only on the overall "vibe" or frequency of the movement.

---

## NLP Models Limitations

1. **Sensor data is not natural language**
   NLP models such as Transformers are designed for token sequences with semantic structure, but the Helios dataset contains continuous sensor readings from IMU, thermopiles, and TOF sensors. Treating sensor windows like “tokens” may lose the physical meaning of motion, temperature, and distance patterns.

2. **Sequence length can make training expensive**
   Natural language processing models, especially those based on attention mechanisms, typically become slower and more memory-intensive as sequence length increases. Since this project uses multi-channel, multivariate time-series data, the memory/GPU resources required for training may be significantly greater than for simple baseline models. The paper itself also points out that deployment constraints such as memory, energy, and storage are critical, and this concern is even more pronounced for larger-scale natural language processing models.

4. **Real-time deployment may be difficult**
   A2 aims to build BFRB detection methods based on Natural Language Processing (NLP), but in practice, these models may be difficult to run on portable or wearable hardware due to issues such as latency, memory, and battery costs. This is especially important for systems that may ultimately require fast feedback or on-device inference.

5. **The models do not explicitly encode spatial structure of TOF data**
   TOF data possesses underlying spatial patterns, but ordinary word sequence NLP models may flatten or simplify this structure. This wastes useful local information, which CNN-based architectures are better at utilizing. The excellent results of CNN-BiLSTM TOF presented in this paper support the view that modality-specific structures are crucial.

6. *Sensor noise and missing values can disrupt token representations**
   If some sensor channels are noisy, unstable, or partially missing, the input representation after word segmentation may become inconsistent. This problem may be more severe for natural language processing (NLP) models, which assume that each lexical unit carries stable structured information.
   
---

## References

1. X. Zhang, J. Ryoo, and S. Mukherjee.  
   *Detection of Body-Focused Repetitive Behaviors using Deep Learning.*  
   2025 IEEE Computing, Communications and IoT Applications Conference (ComComAp 2025), IEEE, 2025.

2. CMI – Detect Behavior with Sensor Data (Kaggle Competition), 2025.  
   Available at: https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data

3. J. W. Cooley and J. W. Tukey.  
   *An Algorithm for the Machine Calculation of Complex Fourier Series.*  
   Mathematics of Computation, 1965.

4. S. Hochreiter and J. Schmidhuber.  
   *Long Short-Term Memory.*  
   Neural Computation, 1997.

5. L. Breiman.  
   *Random Forests.*  
   Machine Learning, 2001.

6. M. S. Ryoo, A. J. Piergiovanni, A. Arnab, M. Dehghani, and A. Angelova.  
   *TokenLearner: What Can 8 Learned Tokens Do for Images and Videos?*  
   Advances in Neural Information Processing Systems (NeurIPS), 2021.

7. A. Vaswani et al.  
   *Attention Is All You Need.*  
   Advances in Neural Information Processing Systems (NeurIPS), 2017.



































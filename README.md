# Plant Disease Classification Using Vision Transformer (ViT)

A deep learning project that fine-tunes a Vision Transformer model to classify plant diseases across 22 categories of Indian plant species. The system distinguishes between healthy and diseased leaf samples, achieving a test accuracy of 94.59%.

Developed by Group 16 as part of the course [25AID111] Introduction to Data Structures and Algorithms at Amrita Vishwa Vidyapeetham.

**Team Members**
- M. Pavan Sai (CB.AI.U4AAR25009)
- N. H. Karthikeya Singh (CB.AI.U4AAR250027)
- K. Srivas Reddy (CB.AI.U4AAR250028)
- B. Dinesh Kumar (CB.AI.U4AAR250033)

---

## Overview

Plant diseases cause significant agricultural loss and early detection is crucial for intervention. This project applies transformer-based image classification to the problem, using the `facebook/deit-small-patch16-224` model fine-tuned on the Plants Disease Dataset. The model was trained on Google Colab using a GPU backend.

The model learns visual features including leaf texture, color patterns, surface discolouration, and disease spots directly from image data.

---

## Dataset

**Name:** Plants Disease Dataset  
**Source:** Google Drive (mounted in Colab)  
**Structure:** Standard ImageFolder layout with `train`, `valid`, and `test` splits

The dataset covers 11 Indian plant species, each with a diseased and healthy class, giving 22 classes total.

| Class Label | Description |
|---|---|
| Alstonia Scholaris diseased (P2a) | Diseased Alstonia Scholaris |
| Alstonia Scholaris healthy (P2b) | Healthy Alstonia Scholaris |
| Arjun diseased (P1a) | Diseased Arjun |
| Arjun healthy (P1b) | Healthy Arjun |
| Bael diseased (P4b) | Diseased Bael |
| Basil healthy (P8) | Healthy Basil |
| Chinar diseased (P11b) | Diseased Chinar |
| Chinar healthy (P11a) | Healthy Chinar |
| Gauva diseased (P3b) | Diseased Gauva |
| Gauva healthy (P3a) | Healthy Gauva |
| Jamun diseased (P5b) | Diseased Jamun |
| Jamun healthy (P5a) | Healthy Jamun |
| Jatropha diseased (P6b) | Diseased Jatropha |
| Jatropha healthy (P6a) | Healthy Jatropha |
| Lemon diseased (P10b) | Diseased Lemon |
| Lemon healthy (P10a) | Healthy Lemon |
| Mango diseased (P0b) | Diseased Mango |
| Mango healthy (P0a) | Healthy Mango |
| Pomegranate diseased (P9b) | Diseased Pomegranate |
| Pomegranate healthy (P9a) | Healthy Pomegranate |
| Pongamia Pinnata diseased (P7b) | Diseased Pongamia Pinnata |
| Pongamia Pinnata healthy (P7a) | Healthy Pongamia Pinnata |

---

## Model Architecture

The project uses **DeiT-Small (Data-efficient Image Transformer)**, a compact Vision Transformer variant from Facebook AI, loaded via HuggingFace Transformers.

- **Base model:** `facebook/deit-small-patch16-224`
- **Input resolution:** 224x224
- **Classification head:** Fine-tuned for 22 output classes
- **Framework:** PyTorch + HuggingFace Transformers

---

## Results

### Training Loss

Training loss decreased steadily over 10 epochs, indicating stable and consistent learning with no signs of divergence.

| Epoch | Training Loss |
|---|---|
| 1 | 1.3553 |
| 2 | 0.2468 |
| 3 | 0.1283 |
| 4 | 0.0762 |
| 5 | 0.0658 |
| 6 | 0.0550 |
| 7 | 0.0387 |
| 8 | 0.0357 |
| 9 | 0.0278 |
| 10 | 0.0163 |

### Validation Accuracy

| Epoch | Validation Accuracy |
|---|---|
| 1 | 86.36% |
| 2 | 90.00% |
| 3 | 92.73% |
| 4 | 90.91% |
| 5 | 93.64% |
| 6 | 89.09% |
| 7 | 91.82% |
| 8 | 93.64% |
| 9 | 94.55% |
| 10 | 93.64% |

Best validation accuracy: **94.55%** (Epoch 9)

### Test Accuracy

**94.59%** on the held-out test set

The confusion matrix shows predictions concentrated along the diagonal, confirming high per-class accuracy. The few misclassifications that occurred were mainly between visually similar healthy and diseased leaf images of the same species.

---

## Requirements

```
torch
torchvision
transformers
scikit-learn
matplotlib
```

To install:

```bash
pip install torch torchvision transformers scikit-learn matplotlib
```

---

## How to Run

This project is designed to run on Google Colab with Google Drive mounted.

1. Upload your dataset to Google Drive under the path `MyDrive/Plants_2/` with `train`, `valid`, and `test` subfolders following the ImageFolder structure.

2. Open the notebook or paste the code from `code/code.txt` into a Colab cell.

3. Run the cells in order. The script will:
   - Mount Google Drive
   - Auto-detect train, validation, and test folders
   - Load and fine-tune the DeiT model
   - Save the best model checkpoint, metrics, graphs, and confusion matrix to `MyDrive/vit_results/`

**Saved outputs include:**
- `best_model.pth` — checkpoint with highest validation accuracy
- `final_model.pth` — model after all epochs
- `metrics.json` — per-epoch loss and accuracy values
- `training_loss.png` — training loss curve
- `validation_accuracy.png` — validation accuracy curve
- `confusion_matrix.png` — test set confusion matrix
- `classes.txt` — list of class names

---

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Batch size | 32 |
| Epochs | 10 |
| Loss function | CrossEntropyLoss |
| Mixed precision | AMP (torch.cuda.amp) |
| Hardware | GPU (CUDA) |

Data augmentation applied during training: random horizontal flip, random rotation (±10 degrees), and resize to 224x224.

---

## Repository Structure

```
.
├── code/
│   └── code.txt             # Full training script
├── results/
│   ├── metrics.json         # Loss and accuracy per epoch
│   ├── classes.txt          # All 22 class names
│   ├── training_loss.png    # Training loss curve
│   ├── validation_accuracy.png
│   └── confusion_matrix.png
└── documents/
    └── ViT_Project_Report.docx
```

---

## Conclusion

The fine-tuned Vision Transformer achieved strong classification performance on the plant disease dataset, with a final test accuracy of 94.59%. Transfer learning from the DeiT-Small checkpoint, combined with data augmentation, allowed the model to generalise well despite the relatively small size of the dataset. The results suggest this approach is a viable foundation for smart farming and automated plant health monitoring applications.

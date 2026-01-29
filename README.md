# blood-cell-classification-CNN-Resnet
Comparative study of blood cell classification using CNN and ResNet50

##  Project Overview
This project presents a comparative study on the classification of blood cell images using deep learning techniques. A baseline Convolutional Neural Network (CNN) is developed and compared against a pretrained ResNet50 model to evaluate performance improvements in multi-class medical image classification.

The goal is to assist automated diagnosis and support laboratory analysis by accurately identifying different blood cell types.

---

##  Objectives
- Classify blood cell images into 8 categories
- Compare baseline CNN performance with pretrained ResNet50
- Apply data augmentation and regularization techniques
- Analyze overfitting and generalization behavior

---

## 🧬 Dataset
- **Source:** Kaggle – Blood Cells Image Dataset  
- **Link:** https://www.kaggle.com/datasets/unclesamulus/blood-cells-image-dataset  
- **Total Images:** ~17,000  
- **Classes:**  
  - Basophil  
  - Eosinophil  
  - Erythroblast  
  - Immature Granulocytes (IG)  
  - Lymphocyte  
  - Monocyte  
  - Neutrophil  
  - Platelet  

Dataset split:
- Training: 70%
- Validation: 15%
- Test: 15%

---

## 🛠️ Tools & Technologies
- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Google Colab

---

##  Models Implemented

### 1️Baseline CNN with Data Augmentation
- Multiple convolutional blocks with Batch Normalization
- Dropout for regularization
- Global Average Pooling
- Softmax output (8 classes)
- Optimizers tested:
  - Adam
  - Learning Rate Scheduler
  - ReduceLROnPlateau

📈 Observations:
- High training accuracy
- Overfitting observed in later epochs
- Improved stability with LR scheduling

---

###  Pretrained ResNet50
- ImageNet pretrained weights
- Frozen convolutional base (initial training)
- Fine-tuning of final layers
- Global Average Pooling + Softmax

📈 Observations:
- Improved generalization
- Slower convergence compared to CNN
- Fine-tuning improved validation accuracy

---

##  Results Summary
| Model | Validation Accuracy |
|------|---------------------|
| Baseline CNN (with LR Scheduler) | ~98% |
| ResNet50 (Fine-tuned) | ~66% |

---

##Related Work
Katar, O., & Yildirim, O. (2023).  
*An Explainable Vision Transformer Model Based White Blood Cells Classification and Localization.*  
Diagnostics, 13(14), 2459.  
https://doi.org/10.3390/diagnostics13142459

---

## How to Run
1. Clone the repository
2. Open the notebook in Jupyter / Colab
3. Upload the dataset or link your Google Drive
4. Run all cells sequentially

---

##  Author
**Shivani Boppala**  
Master’s Student – Data Analysis
Aspiring Data Analyst 

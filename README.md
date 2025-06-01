# 🧠 Concept Detection in Radiology Images (ROCOv2)

This repository contains the code and resources used for the **Concept Detection Task** of the **ImageCLEFmedical Caption 2025** challenge. The objective is to identify UMLS concepts from radiology images using deep learning models.

## 📌 Project Summary

We developed and evaluated several convolutional neural network (CNN) models including ResNet50, DenseNet121, and InceptionV3 to predict medical concepts (CUIs) from radiology images. Our best results were achieved using a pretrained ResNet50 backbone, fine-tuned on the ROCOv2 dataset using a custom multi-label classification head.The trained model can be accessed in the following link https://drive.google.com/file/d/1eLV_x1prRa0Oal94UI1HZdq7aLCGiEjW/view?usp=drive_link.

---

## 🧪 Tasks Performed

- Preprocessed radiology images and UMLS labels
- Applied one-hot/multi-hot encoding using `MultiLabelBinarizer`
- Designed a custom image data generator for multi-label tasks
- Implemented transfer learning with:
  - ResNet50 (RadImageNet pretrained)
  - InceptionV3
  - DenseNet121
- Trained models on 80K+ training images, validated on ~17K images
- Achieved best validation accuracy of **61.57%** with a loss of **0.1370** using top-10 frequent CUIs

---

## 🚀 Model Architecture

- **Backbone**: ResNet50 (without top layers), pretrained on RadImageNet
- **Classifier**: Global Average Pooling → BatchNorm → Dense (512→256→128→64) → Output (10) with sigmoid
- **Loss**: Binary Crossentropy
- **Optimizer**: Adam
- **EarlyStopping**: Enabled

---

## 🔄 Preprocessing Pipeline

1. Load dataframe and images
2. Extract top 10 most frequent CUIs
3. Encode labels with `MultiLabelBinarizer`
4. Use a custom data generator (`multilabel_flow_from_dataframe`)
5. Normalize images to (224, 224, 3)

---

## 📈 Results

| Model         | Top-K CUIs | Validation Accuracy | Val Loss |
|---------------|------------|---------------------|----------|
| ResNet50      | 10         | **61.57%**          | 0.1370   |
| DenseNet121   | 10         | ~56%                | -        |
| InceptionV3   | 10         | ~54%                | -        |

- **Best performance** was obtained with ResNet50 using pretrained weights from RadImageNet.

---

## 🔍 Result Analysis

- ResNet50 outperformed other architectures likely due to:
  - Compatibility with medical features via RadImageNet pretraining
  - Reduced overfitting via Dropout and BatchNorm
- InceptionV3 and DenseNet121 showed higher training loss and slower convergence
- Lower test set performance may be due to:
  - Label imbalance
  - Test data domain shift
  - Limited representation with only top-10 labels

---

## 🧠 Future Work

- Use hierarchical classification with full UMLS semantic types
- Apply attention mechanisms (e.g., CBAM, SE-blocks)
- Use medical captioning to jointly train with concept detection
- Incorporate bounding box or segmentation data where available

---

## 🛠️ Resources Employed

- Mixed environments:
  - Kaggle Notebooks (TPU/V100)
  - University GPU clusters (RTX 3090)
- Frameworks:
  - TensorFlow / Keras
  - Pandas, NumPy, scikit-learn
- Datasets:
  - ROCOv2 (Radiology Objects in Context)

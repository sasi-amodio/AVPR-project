# **Image Classification and Evaluation with CNN, LBP, HoG, and ULDP**

This project demonstrates **image classification** and **evaluation** using various feature extraction methods such as **Convolutional Neural Networks (CNN)**, **Local Binary Patterns (LBP)**, **Histogram of Oriented Gradients (HoG)**, and **Universal Local Directional Patterns (ULDP)**. The goal is to classify images into different categories and evaluate the performance of each method using **Support Vector Machines (SVM)**.

## **Project Overview**

This project provides an end-to-end pipeline for image classification and evaluation. It utilizes several feature extraction techniques (CNN, LBP, HoG, ULDP) and compares their performance using machine learning algorithms (SVM). The main goal is to evaluate the ability of these techniques to classify images from a custom dataset.

The project includes the following components:
- **Data Import**: Loading and organizing training and test data.
- **Feature Extraction**: Using CNN, LBP, HoG, and ULDP to extract features from images.
- **Model Training**: Training SVM models on the extracted features.
- **Model Evaluation**: Evaluating performance using ROC curves, AUC, and confusion matrices.

## **Features**

- **Data Preprocessing**: Easily import and organize your dataset for training and testing.
- **Multiple Feature Extraction Methods**: Support for CNN, LBP, HoG, and ULDP.
- **Model Training**: SVM classifier using an RBF kernel to classify images.
- **Evaluation**: ROC curves, AUC scores, and confusion matrix to assess the model's performance.

## **Installation**

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/project-repo.git

# chatbot_intern_codec
End-to-End AI Chatbot built entirely in Google Colab. Features a full NLP pipeline comparing traditional Machine Learning models against a Deep Learning (Bi-LSTM) architecture. Developed as a 2-Month AI Internship Project.
# 🤖 SmartAI Chatbot: Machine Learning vs. Deep Learning Pipeline

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-yellow.svg)
![Google Colab](https://img.shields.io/badge/Google_Colab-Ready-f9ab00.svg)

## 📌 Project Overview
This repository contains a comprehensive Artificial Intelligence Chatbot developed as a **2-Month AI Internship Project**. The project is designed to run entirely within a single Google Colab notebook, demonstrating a complete end-to-end NLP and AI engineering workflow. 

Instead of a simple rule-based bot, this project implements a full data science pipeline, comparing multiple traditional Machine Learning algorithms against a modern Deep Learning architecture (Bidirectional LSTM) for intent classification.

## 🚀 Key Features
* **Automated Data Generation:** Programmatic creation of a 1,000+ sample training dataset spanning 40+ distinct conversational intents.
* **Robust NLP Preprocessing:** Implementation of tokenization, lemmatization, stop-word removal, and TF-IDF vectorization.
* **Multi-Model Machine Learning:** Training and evaluation of Logistic Regression, Naive Bayes, Decision Trees, Random Forests, and SVMs.
* **Deep Learning Architecture:** A custom TensorFlow/Keras neural network featuring an Embedding Layer, Bidirectional LSTM, and Dropout layers.
* **Automated Benchmarking:** Direct comparison of ML vs. DL models based on Accuracy, Precision, Recall, F1-Score, and inference speed.
* **Interactive Inference Loop:** A built-in terminal chatbot interface featuring confidence scoring, context awareness, and fallback intent routing.

## 🛠️ Technology Stack
* **Language:** Python
* **Deep Learning:** TensorFlow / Keras
* **Machine Learning:** Scikit-Learn
* **NLP Processing:** NLTK
* **Data Manipulation:** NumPy, Pandas
* **Data Visualization:** Matplotlib, Seaborn
* **Environment:** Google Colab

## 🧠 Deep Learning Architecture (Bi-LSTM)
The custom neural network designed for this project consists of:
1. **Input Layer:** Tokenized and padded sequences.
2. **Embedding Layer:** Dense vector representations of words.
3. **Bidirectional LSTM:** Captures both past and future contextual dependencies in user queries.
4. **Dropout Layers:** Prevents overfitting during training.
5. **Dense (ReLU):** Fully connected hidden layer for deep feature extraction.
6. **Softmax Output:** Multi-class probability distribution for intent classification.

## 💻 How to Run the Project
This project requires **zero local setup**. It is designed to execute seamlessly in the cloud.

1. Click the file: `SmartAI_Chatbot_Project.ipynb`
2. Click **"Open in Colab"** at the top of the file preview.
3. In Google Colab, navigate to the top menu and select **Runtime > Run all**.
4. The notebook will automatically install dependencies, generate the dataset, train all models, output performance graphs, and launch the interactive chatbot at the very bottom of the notebook.

## 📊 Model Evaluation Visualizations
The notebook automatically generates diagnostic charts, including:
* Dataset Intent Distribution
* Cross-Model Accuracy Comparison Matrix
* Deep Learning Training/Validation Accuracy & Loss Curves
* Confusion Matrices for misclassification tracking

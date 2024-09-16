# Mood Machine Learning Recommender System

## Overview

The Mood Machine Learning Recommender System is designed to predict and recommend songs based on their features and mood. This project leverages various machine learning models to classify songs into different moods and provides song recommendations accordingly.

## Project Structure

### 1. Model Training and Evaluation

- **File**: `train_evaluate_models.py`
- **Description**: This script trains and evaluates multiple machine learning models including Logistic Regression, SGD Classifier, Naive Bayes, Decision Tree, Random Forest, XGBoost, SVM, and KNN.
- **Features**:
  - `danceability`
  - `energy`
  - `valence`
  - `tempo`
  - `acousticness`
  - `instrumentalness`
  - `liveness`
  - `loudness`
  - `speechiness`
  - `key` (one-hot encoded)
- **Target**: `mood` (e.g., Calm, Excited, Happy, etc.)

### 2. Hyperparameter Tuning

- **File**: `KNNv1.ipynb`
- **Description**: Performs hyperparameter tuning for the Random Forest model using GridSearchCV to find the best parameters and evaluate its performance.
- **Hyperparameters Tuned**:
  - `n_estimators`
  - `max_depth`
  - `min_samples_split`
  - `min_samples_leaf`
  - `max_features`

	## Learning Curve of a model
	![learning curve](https://github.com/user-attachments/assets/b5a1c3d7-bbf6-4877-9287-d58e6bf0c0e7)

### 3. Song Recommendation

- **File**: `MoodML.py`
- **Description**: Provides song recommendations based on a specified mood using the best-trained Random Forest model.
- **Features**:
  - `explicit`
  - `danceability`
  - `energy`
  - `key`
  - `loudness`
  - `mode`
  - `speechiness`
  - `acousticness`
  - `instrumentalness`
  - `liveness`
  - `valence`
  - `tempo`
  - `duration_ms`

## Setup

1. **Install Dependencies**

     Ensure you have the necessary Python packages installed. You can install them using pip:
     `pip install pandas scikit-learn xgboost`

2. **Prepare the Dataset**

    - `songs_with_moods_ML_Dataset.csv`: Contains song features and associated moods for recommendation.
    - `songs_with_moods.csv`: Used for training and evaluating models
    ![variation](https://github.com/user-attachments/assets/f1570e18-4f23-4217-9493-897839ba5c37)

    

3. **Load and Train Models**
    
    Run the `KNNv1.ipynb` script to train and evaluate the machine learning models:
    
4. **Tune Hyperparameters**
    
    Run the `KNNv1.ipynb` script to tune the Random Forest model:
    
5. **Predict and Recommend Songs**
    
    Use the ``MoodML.py`` script to get song recommendations based on mood:
    
    `python MoodML.py`
    

## Usage

### Model Training

The ``KNNv1.ipynb`` script trains several models and evaluates their performance using metrics such as accuracy, precision, recall, and F1 score. It outputs the performance metrics of each model.

### Hyperparameter Tuning

The `KNNv1.ipynb` script performs grid search to find the best hyperparameters for the Random Forest model and evaluates its performance with the tuned parameters.

### Song Recommendation

The `MoodML.py` script takes a mood input and returns recommended songs by using the trained Random Forest model to filter and sample songs from the dataset.

## Example

To get recommendations for the mood "Happy":

`from recommend_songs import predict_songs
mood_input = 'Happy'
recommendations = predict_songs(mood_input)
print(recommendations)`

## Files

- `KNNv1.ipynb`:    Script for training and evaluating various machine learning models.
- `KNNv1.ipynb`:    Script for tuning hyperparameters of the Random Forest model.
- `MoodML.py`:        Script for recommending songs based on mood.
- `songs_with_moods.csv`:   Dataset for training and evaluating models.
- `songs_with_moods_ML_Dataset.csv`:   Dataset for song recommendations.

# Cold-Start Recommendation System 

## Project Overview
The project focuses on developing a Recommender System by exploring different models to improve accuracy and performance, including a Neural Network Cold-Start Solver, Matrix Factorization with an enhanced approach, and Neural Collaborative Filtering (NCF) to capture complex user-item interactions.

- [Neural Network Cold Start Solver](#neural-network-cold-start-solver)
- [Matrix Factorization with an enhanced approach](#matrix-factorization-with-an-enhanced-approach)
- [Neural Collaborative Filtering (NCF)](#neural-collaborative-filtering)



## Neural Network Cold Start Solver
## Dataset Description

### Input Data Format
For each user, the dataset contains sequences of product interactions and their respective ratings:
- **User**: `u1 → [{p1, r1}, {p2, r2}, {p3, r3}, ..., {pn, rn}]`

### Model Input
- **Product IDs**: The first three products in the sequence (`p1`, `p2`, `p3`).
- **Ratings**: The corresponding ratings for the first three products (`r1`, `r2`, `r3`).
- **Dislike Flag**: A binary flag (`1` for disliked, `0` for liked) derived from a threshold (e.g., dislike if rating ≤ 3).

### Model Output
- Predicted ratings vector: `[r1, r2, r3, ..., rk]` without prior knowledge of future ratings (`r4, ..., rk`).

---

## Experimentation and Results

### Models Evaluated
1. **LSTM Variants**:
   - **LSTM1**: A simple LSTM-based model for sequential prediction.
   - **LSTM2**: An enhanced LSTM model that incorporates Conv1D layers, dislike flags, and richer feature combinations.
   
2. **GRU**:  
   - A Gated Recurrent Unit (GRU)-based model for sequence modeling.
   
3. **RNN**:  
   - A simpler Recurrent Neural Network model to capture sequential dependencies.

4. **Ensemble**:  
   - A hybrid approach combining multiple architectures to boost performance.

5. **Baseline Models**:
   - **BERT4Rec**: A transformer-based model designed for recommendations.
   - **GRU4Rec**: A GRU-based model tailored for recommendation tasks.

---

### Performance Metrics

| Model        | RMSE   | MAE    |
|--------------|--------|--------|
| **LSTM1**    | 0.86692 | 0.57565 |
| **LSTM2**    | **0.86321** | **0.57957** |
| **RNN**      | 0.97540 | 0.70417 |
| **GRU**      | 0.91682 | 0.67359 |
| **Ensemble** | **0.83624** | **0.58670** |
| **BERT4Rec*** | 1.0907  | 0.7971  |
| **GRU4Rec***  | 1.0794  | 0.7797  |
| **GRU4Rec MELO***  | 0.9857  | 0.67  |
| **BERT4Rec MELO***  | 0.9925 | 0.6608  |

---
* presents Baseline SOTA Models.
Our Best Performing Model is the Ensemble and Best Performing Baseline model is GRU4Rec MELO.
Our Model Improves over basic by **15.88%**

### Observations

1. **Best Performing Model**:
   - `LSTM2` demonstrated strong performance with an RMSE of **0.86321** and MAE of **0.57957**.
   - The use of Conv1D layers, dislike flags, and LSTM architecture improved the model’s ability to learn patterns in user behavior.

2. **Ensemble Model**:
   - The ensemble approach achieved the lowest RMSE (**0.83624**), but its MAE was slightly higher than that of `LSTM2`.

3. **Baseline Models**:
   - `BERT4Rec` and `GRU4Rec` underperformed in this scenario, likely due to their dependence on longer sequences or pretraining.

4. **Simpler Models**:
   - `RNN` and `GRU` were less effective at capturing complex patterns, resulting in higher RMSE and MAE scores.

---

## Key Features

- **Sequence Modeling**: Predicts future ratings based on a sequence of three product interactions.
- **Dislike Flag Integration**: Helps the model recognize patterns in user preferences, improving prediction accuracy for disliked products.
- **Custom Loss Functions**:
  - `Masked MSE`: Excludes invalid predictions from loss calculation.
  - `Masked MAE` and `Masked RMSE`: Focus on valid predictions during evaluation.

---

## Conclusion

The project demonstrated that combining sequence modeling with auxiliary features like dislike flags effectively addresses the cold-start problem. The **LSTM2 model** achieved the best balance of RMSE and MAE, outperforming state-of-the-art baselines in this cold-start scenario. Future work could involve fine-tuning the architecture and exploring longer sequences for enhanced performance.

## Matrix Factorization with an enhanced approach
This project implements a recommendation system using matrix factorization. Below is an overview of the steps taken to build and evaluate the model.

## Libraries Imported
- pandas
- numpy
- seaborn
- matplotlib

## Data Loading and Exploration
1. Loaded a CSV file (ratings.csv) into a DataFrame.
2. Displayed the first few rows (head) and information about the DataFrame (info).

## Data Preprocessing
1. Dropped the timestamp column.
2. Checked for missing values and removed rows with NaN values.
3. Sorted the data by movieId and added a Count column to track the popularity of movies.
4. Filtered movies with a popularity_threshold of 1000.

## Transformation
1. Converted the data into a pivot table with users as rows and movies as columns.
2. Renamed columns to include a "movie" prefix.
3. Filled NaN values with 0.

## Matrix Factorization
1. Defined a custom function for matrix factorization.
2. Computed the rank of the ratings matrix and used it for factorization.
3. Factorized the matrix into:
   - U: User feature matrix.
   - V: Movie feature matrix.

## Reconstruction and Evaluation
1. Reconstructed the ratings matrix using U and V.
2. Compared the original matrix with the reconstructed one using cosine similarity.

## Model Accuracy
- Calculated accuracy by checking how close the predicted values are to the original values within a specified margin.

## Extended Testing
1. Split the data into training and test sets.
2. Predicted ratings for the test set and evaluated accuracy.

---
## Performance Over Existing Arch:
Collaborative Filtering RMSE: 1.71
Matrix Factorization RMSE :1.30

## Neural Collaborative Filtering

*Neural Collaborative Filtering (NCF)*, a deep learning-based recommendation system that improves the accuracy of predictions by modeling complex, non-linear user-item interactions.

## Overview

The NCF model utilizes a *Multi-Layer Perceptron (MLP)* to learn and predict interactions between users and products. The model is designed to handle sparse data and cold-start problems effectively by incorporating embedding layers for users and items. A sequence modeling technique with a dislike flag is also used to further improve recommendation quality, especially for new or less active users.

## Key Features

- *User and Item Embeddings*: Low-dimensional representations of users and items learned during training.
- *Multi-Layer Perceptron (MLP)*: Captures non-linear user-item interactions.
- *Cold-Start Solver: Uses sequence modeling with a length of three interactions and a *dislike flag to improve recommendations for new users.
- *Scalability*: The model is designed to handle large-scale datasets efficiently.
  
## Experimental Settings

- *Dataset: The model was trained on the **Amazon Electronics dataset*, which contains user-item interaction data.
- *Evaluation Metrics*:
  - *Root Mean Squared Error (RMSE)*
  
- *Hyperparameters*:
  - *Batch Size*: 64
  - *Epochs*: 10
  - *Optimizer*: Adam
  - *Learning Rate*: 0.001
  
  
## Model Architecture

1. *Embedding Layers*: Represent users and items in a lower-dimensional space.
2. *Multi-Layer Perceptron (MLP)*: Learn non-linear interactions between the user and item embeddings.
3. *Prediction Layer*: Outputs the predicted rating or interaction probability.

## Performance Over Existing Architecture:
Collaborative Filtering RMSE: 1.71
NCF RMSE :1.039

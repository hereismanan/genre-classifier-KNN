
# Music Genre Classifier (KNN)

This project uses the K-Nearest Neighbors (KNN) algorithm to classify music genres based on extracted audio features.

## Features

  * **Audio Feature Extraction**: Extracts MFCCs, Chroma, Spectral Contrast, Spectral Rolloff, Zero-Crossing Rate, RMS, and Tempo using `librosa`.
  * **KNN Classification**: Employs a K-Nearest Neighbors model for genre prediction.
  * **Data Preprocessing**: Includes feature scaling and label encoding.
  * **Hyperparameter Tuning**: Utilizes `GridSearchCV` to optimize KNN parameters.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/hereismanan/genre-classifier-knn.git
    cd genre-classifier-knn
    ```
2.  Install dependencies:
    ```bash
    pip install numpy pandas matplotlib seaborn librosa scikit-learn tqdm
    ```

## Usage

1.  Organize your audio files by genre into `Data/genres_original/` (e.g., `Data/genres_original/blues/song.wav`).
2.  Run the `Genre-Classifier-KNN.ipynb` Jupyter Notebook to extract features, train the model, and evaluate its performance.

## Results

  * **Initial KNN Accuracy**: 50.5%
  * **Best Hyperparameters (tuned with GridSearchCV)**: `{'metric': 'manhattan', 'n_neighbors': 9}`
  * **Final Test Accuracy**: 63.5%

-----

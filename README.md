# IMDB Movie Review Sentiment Analysis Using LSTM

## Overview

This project implements a Long Short-Term Memory (LSTM) neural network for binary sentiment classification of movie reviews using TensorFlow and Keras.

The model is trained on the IMDB movie review dataset to classify reviews as positive or negative. It demonstrates an end-to-end NLP workflow, including loading encoded text data, decoding reviews, sequence padding, embedding generation, model training, evaluation, and sentiment prediction.

## Objectives

- Understand sentiment analysis using deep learning.
- Work with the built-in IMDB movie review dataset.
- Decode numerical review sequences into readable text.
- Apply sequence padding to standardize input lengths.
- Learn word representations through an Embedding layer.
- Build an LSTM-based binary classification model.
- Evaluate model performance on a separate test dataset.
- Predict sentiment probabilities for individual movie reviews.

## Technologies Used

- Python 3.12
- TensorFlow and Keras
- IMDB movie review dataset

## Dataset

The project uses the IMDB dataset provided by `tensorflow.keras.datasets.imdb`.

The dataset contains 50,000 movie reviews divided into:

- **25,000 training reviews**
- **25,000 testing reviews**

Each review has a binary sentiment label:

- `0` — Negative
- `1` — Positive

The implementation limits the vocabulary to the 10,000 most frequent words and uses the first 20% of the training data, as determined by Keras's `validation_split`, for validation.

## Model Architecture

The model consists of three main layers:

1. **Embedding Layer:** Converts token IDs into dense vectors of 32 dimensions.
2. **LSTM Layer:** Processes sequential information using 64 LSTM units to learn patterns in movie-review text.
3. **Dense Output Layer:** Uses one neuron with sigmoid activation to produce a probability for binary sentiment classification.

### Model Configuration

| Parameter | Value |
|---|---|
| Vocabulary size | 10,000 |
| Maximum sequence length | 200 |
| Embedding dimension | 32 |
| LSTM units | 64 |
| Output units | 1 |
| Optimizer | Adam |
| Loss function | Binary cross-entropy |
| Evaluation metric | Accuracy |
| Training epochs | 3 |
| Batch size | 64 |
| Validation split | 20% |

## Workflow

1. Import the required TensorFlow and Keras components.
2. Load the IMDB training and testing datasets.
3. Retrieve the word-index dictionary.
4. Build a reverse dictionary to decode encoded reviews.
5. Display sample reviews with their actual sentiment labels.
6. Pad training and testing sequences to a fixed length of 200.
7. Construct the Embedding, LSTM, and Dense layers.
8. Compile the model using Adam and binary cross-entropy.
9. Train the model for three epochs with a validation split.
10. Evaluate the model using the test dataset.
11. Decode a selected test review and display its actual sentiment.
12. Predict the review's sentiment and display the predicted probability.

## Installation and Execution

### Prerequisites

Install Python 3.12 (64-bit).

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd IMDB-Movie-Review-Sentiment-Analysis-LSTM
```

Replace the placeholder with your actual repository URL.

### 2. Create a virtual environment

```bash
py -3.12 -m venv venv
```

### 3. Activate the environment on Windows

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the project

```bash
python 3_LSTMModel.py
```

The IMDB dataset is downloaded automatically by Keras when it is not already available locally. An internet connection is required for the initial download.

## Output

The script displays:

- Number of training and testing reviews.
- Sample decoded movie reviews and their actual labels.
- Padded training and testing data shapes.
- Model training progress and validation metrics.
- Test-set evaluation results.
- The selected review's actual sentiment.
- The predicted sentiment probability and classification.

The final classification uses a threshold of 0.5:

- Probability greater than or equal to 0.5: **Positive**
- Probability below 0.5: **Negative**

## Limitations

- The model is trained for three epochs, so additional experimentation may improve its performance.
- The final prediction demonstrates inference on one selected test review rather than a custom user-entered review.
- Performance should be reported using the actual evaluation results produced by the script.
- The model is an educational implementation and has not been established as suitable for production deployment.

## Future Improvements

- Experiment with additional training epochs and hyperparameter tuning.
- Compare LSTM with SimpleRNN and GRU architectures.
- Add a function to accept and classify custom movie reviews.
- Generate a confusion matrix and classification report.
- Save and reload the trained model for reuse.
- Experiment with dropout and other regularization techniques.

## Author

**Aarti Wamane**

GitHub: https://github.com/aartiwamane

LinkedIn: https://www.linkedin.com/in/aarti-wamane-5276562a6/


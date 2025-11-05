# CNN Text Classification

A comprehensive learning project demonstrating how to build Convolutional Neural Networks (CNNs) for text classification tasks using TensorFlow and Keras.

## Overview

This repository contains a detailed Jupyter notebook that teaches you how to implement CNN-based text classification from scratch. While CNNs are primarily known for image processing, they are remarkably effective for text classification tasks due to their ability to detect local patterns (n-grams) and their computational efficiency.

### What You'll Learn

- **CNN Architecture for Text**: Understanding how 1D convolutions work on sequential text data
- **Text Preprocessing**: Tokenization, padding, and sequence preparation
- **Word Embeddings**: Converting text to dense vector representations
- **Multiple CNN Architectures**:
  - Simple CNN with single filter size
  - Multi-filter CNN (Kim 2014 architecture) for capturing different n-gram patterns
- **Training Best Practices**: Early stopping, model checkpointing, and regularization
- **Model Evaluation**: Comprehensive metrics, confusion matrices, and error analysis
- **Making Predictions**: Deploying the model on custom text

## Features

- Complete implementation of CNN text classifiers
- Detailed explanations and comments throughout
- Visualization of training progress, embeddings, and results
- Comparison between different CNN architectures
- Custom text prediction functionality
- Model saving and loading demonstrations
- Analysis of misclassified examples
- Interactive exercises and challenges

## Project Structure

```
CNN-Text-Classification/
│
├── cnn_text_classification.ipynb    # Main learning notebook
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── models/                           # Saved models (created during training)
│   ├── best_model.keras
│   └── cnn_text_classifier.keras
└── data/                            # Dataset directory (auto-downloaded)
```

## Requirements

### Software Requirements

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- pip package manager

### Python Dependencies

All required packages are listed in `requirements.txt`. Key dependencies include:

- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- NLTK

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/CNN-Text-Classification.git
cd CNN-Text-Classification
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook cnn_text_classification.ipynb
```

Or if you prefer JupyterLab:

```bash
jupyter lab cnn_text_classification.ipynb
```

## Quick Start

1. **Install dependencies** as described above
2. **Open the notebook** in Jupyter
3. **Run all cells** sequentially (Cell → Run All)
4. **Experiment** with different parameters and architectures

The notebook uses the IMDB movie reviews dataset, which will be automatically downloaded when you run the data loading cells.

## Dataset

The project uses the **IMDB Movie Reviews Dataset** for binary sentiment classification:

- **Training samples**: 25,000 movie reviews
- **Test samples**: 25,000 movie reviews
- **Classes**: Positive (1) and Negative (0) sentiment
- **Format**: Preprocessed and encoded text sequences

The dataset is automatically downloaded via `tensorflow.keras.datasets.imdb`.

### Using Your Own Dataset

To use a custom dataset, modify the data loading section in the notebook:

```python
# Your custom data loading code
texts = [...]  # List of text strings
labels = [...]  # List of labels

# Tokenization
tokenizer = Tokenizer(num_words=MAX_FEATURES)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=MAX_LEN)
```

## Model Architectures

### 1. Simple CNN

A straightforward CNN architecture with:
- Embedding layer (128 dimensions)
- Single Conv1D layer (128 filters, kernel size 3)
- Global max pooling
- Dense classification layer

**Parameters**: ~1.3M trainable parameters

### 2. Multi-Filter CNN (Kim 2014)

Advanced architecture using multiple filter sizes:
- Embedding layer (128 dimensions)
- Three parallel Conv1D branches (filters sizes: 3, 4, 5)
- Concatenation of features
- Dense classification layer

**Parameters**: ~1.4M trainable parameters

This architecture captures different n-gram patterns simultaneously, leading to better performance.

## Results

Expected performance on IMDB dataset:

| Model | Test Accuracy | Test Loss |
|-------|---------------|-----------|
| Simple CNN | ~86-87% | ~0.35 |
| Multi-Filter CNN | ~87-89% | ~0.32 |

*Note: Results may vary slightly due to random initialization*

## Usage Examples

### Training the Model

```python
# Build model
model = build_multi_filter_cnn(
    vocab_size=10000,
    embedding_dim=128,
    max_length=500,
    num_filters=128,
    filter_sizes=[3, 4, 5],
    dropout_rate=0.5
)

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=20,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, model_checkpoint]
)
```

### Making Predictions

```python
# Predict sentiment for custom text
review = "This movie was absolutely fantastic!"
sentiment, confidence, score = predict_sentiment(
    review, model, word_index, max_len=500
)

print(f"Sentiment: {sentiment}")
print(f"Confidence: {confidence:.2%}")
```

### Loading a Saved Model

```python
# Load model
loaded_model = tf.keras.models.load_model('cnn_text_classifier.keras')

# Make predictions
predictions = loaded_model.predict(X_test)
```

## Key Concepts

### Why CNN for Text?

1. **Local Pattern Detection**: Convolution filters can identify important n-gram patterns
2. **Parameter Efficiency**: Fewer parameters than RNNs for comparable performance
3. **Parallelization**: Faster training compared to sequential models like LSTMs
4. **Translation Invariance**: Features are detected regardless of position

### CNN Components for Text

1. **Embedding Layer**: Maps word indices to dense vectors
2. **Conv1D Layer**: Applies filters to detect local patterns
3. **Pooling Layer**: Extracts most salient features
4. **Dense Layer**: Performs final classification

## Hyperparameter Tuning

Key hyperparameters to experiment with:

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `MAX_FEATURES` | Vocabulary size | 5,000 - 20,000 |
| `MAX_LEN` | Sequence length | 200 - 500 |
| `EMBEDDING_DIM` | Embedding dimensions | 50 - 300 |
| `NUM_FILTERS` | Number of filters per size | 64 - 256 |
| `FILTER_SIZES` | Convolution kernel sizes | [2,3,4], [3,4,5] |
| `DROPOUT_RATE` | Dropout probability | 0.3 - 0.6 |

## Advanced Topics

The notebook covers several advanced topics:

- **Multi-filter architectures** for capturing various n-gram patterns
- **Activation visualization** to understand what the model learns
- **Error analysis** to identify model weaknesses
- **Prediction confidence** analysis
- **Model comparison** between architectures

## Common Issues and Solutions

### Issue: Out of Memory Error

**Solution**: Reduce batch size or sequence length
```python
BATCH_SIZE = 64  # Instead of 128
MAX_LEN = 300    # Instead of 500
```

### Issue: Model Overfitting

**Solution**: Increase dropout or add regularization
```python
DROPOUT_RATE = 0.6  # Increase dropout
# Or add L2 regularization to Dense layers
Dense(128, activation='relu', kernel_regularizer=l2(0.01))
```

### Issue: Poor Performance

**Solutions**:
- Increase vocabulary size (`MAX_FEATURES`)
- Use pre-trained embeddings (GloVe, Word2Vec)
- Add more filters or filter sizes
- Train for more epochs
- Ensure data is properly preprocessed

## Performance Optimization

### Training Speed

- Use GPU acceleration if available
- Increase batch size (if memory allows)
- Reduce sequence length for faster experimentation

### Model Accuracy

- Use pre-trained embeddings
- Experiment with different filter combinations
- Try deeper architectures
- Implement data augmentation
- Use ensemble methods

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Areas for Contribution

- Additional model architectures
- Support for multi-class classification
- Pre-trained embedding integration
- More datasets and examples
- Performance optimizations
- Documentation improvements

## References

### Papers

1. **Kim, Y. (2014)**. "Convolutional Neural Networks for Sentence Classification"
   - Original paper introducing CNN for text classification
   - [ArXiv Link](https://arxiv.org/abs/1408.5882)

2. **Zhang, Y., & Wallace, B. (2015)**. "A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification"
   - Comprehensive analysis of CNN hyperparameters
   - [ArXiv Link](https://arxiv.org/abs/1510.03820)

### Tutorials and Resources

- [Understanding CNNs for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)
- [TensorFlow Text Classification Tutorial](https://www.tensorflow.org/tutorials/keras/text_classification)
- [Deep Learning for NLP Best Practices](http://ruder.io/deep-learning-nlp-best-practices/)

### Datasets

- [IMDB Reviews](https://ai.stanford.edu/~amaas/data/sentiment/)
- [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/)
- [AG News](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow and Keras teams for excellent documentation
- Yoon Kim for the original CNN text classification paper
- Stanford for the IMDB dataset
- The open-source community for various tools and libraries

## Contact

For questions, issues, or suggestions, please:
- Open an issue on GitHub
- Submit a pull request
- Contact the maintainer

## Citation

If you use this code in your research or projects, please cite:

```bibtex
@misc{cnn_text_classification,
  author = {Your Name},
  title = {CNN Text Classification: A Learning Project},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/CNN-Text-Classification}
}
```

---

**Happy Learning!** If you find this project helpful, please consider giving it a star on GitHub.
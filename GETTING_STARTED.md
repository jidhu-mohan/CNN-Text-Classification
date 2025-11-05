# Getting Started Guide

This guide will help you get up and running with the CNN Text Classification project in just a few minutes.

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

## Step-by-Step Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/jidhu-mohan/CNN-Text-Classification.git
cd CNN-Text-Classification
```

### Step 2: Create a Virtual Environment

It's recommended to use a virtual environment to avoid conflicts with other projects.

**Option A: Using venv (built-in)**
```bash
python -m venv venv

# Activate on macOS/Linux:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

**Option B: Using conda**
```bash
conda create -n cnn-text python=3.10
conda activate cnn-text
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all necessary packages including:
- TensorFlow/Keras for deep learning
- NumPy and Pandas for data processing
- Matplotlib and Seaborn for visualization
- scikit-learn for metrics
- NLTK for text processing
- Jupyter for running notebooks

### Step 4: Verify Installation

Run a quick check to ensure everything is installed correctly:

```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import numpy as np; print('NumPy version:', np.__version__)"
python -c "import pandas as pd; print('Pandas version:', pd.__version__)"
```

### Step 5: Launch Jupyter Notebook

```bash
jupyter notebook cnn_text_classification.ipynb
```

This will open the notebook in your default web browser.

Alternatively, if you prefer JupyterLab:

```bash
jupyter lab cnn_text_classification.ipynb
```

## Running Your First Model

Once the notebook is open:

1. **Execute cells sequentially**: Start from the top and run each cell in order
   - Use `Shift + Enter` to run a cell and move to the next one
   - Or use `Cell > Run All` to run all cells

2. **Wait for data download**: The first time you run, the IMDB dataset will be downloaded automatically (this may take a few minutes)

3. **Monitor training**: Watch the training progress bars and accuracy metrics

4. **Explore results**: Check the visualizations, confusion matrix, and prediction examples

## Quick Tips

### For Faster Experimentation

If you want to quickly test the code without waiting for full training:

1. Reduce the number of epochs:
   ```python
   EPOCHS = 5  # Instead of 20
   ```

2. Use a smaller dataset:
   ```python
   # Reduce training size
   X_train = X_train[:5000]
   y_train = y_train[:5000]
   ```

3. Reduce sequence length:
   ```python
   MAX_LEN = 200  # Instead of 500
   ```

### For Better Performance

If you have a GPU available:

1. Verify GPU is detected:
   ```python
   print("GPUs Available:", tf.config.list_physical_devices('GPU'))
   ```

2. TensorFlow should automatically use the GPU

3. If needed, install GPU-specific TensorFlow:
   ```bash
   pip install tensorflow-gpu
   ```

### Common First-Time Issues

**Issue**: Jupyter kernel not found
```bash
python -m ipykernel install --user --name=cnn-text
```

**Issue**: NLTK data missing
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

**Issue**: Out of memory during training
- Reduce batch size: `BATCH_SIZE = 64`
- Reduce sequence length: `MAX_LEN = 300`
- Close other applications

## What to Explore

### 1. Modify Hyperparameters

Try changing these values to see how they affect performance:
- `EMBEDDING_DIM`: Size of word embeddings
- `NUM_FILTERS`: Number of convolutional filters
- `FILTER_SIZES`: Different n-gram sizes to capture
- `DROPOUT_RATE`: Regularization strength

### 2. Try Different Architectures

The notebook includes:
- Simple CNN with single filter size
- Multi-filter CNN (Kim 2014 architecture)

Compare their performance on the same data.

### 3. Use Your Own Data

Replace the IMDB dataset with your own text data:
```python
# Load your custom data
texts = ["your text here", "another text", ...]
labels = [0, 1, ...]  # Your labels

# Follow the preprocessing steps in the notebook
```

### 4. Experiment with Predictions

Try the model on different types of text:
- Movie reviews
- Product reviews
- Tweets
- News articles

## Next Steps

After completing the basic tutorial:

1. **Read the Full README**: [README.md](README.md) for detailed documentation

2. **Explore Advanced Topics**:
   - Implement attention mechanisms
   - Try pre-trained embeddings (GloVe, Word2Vec)
   - Experiment with deeper architectures
   - Add data augmentation techniques

3. **Apply to Real Projects**:
   - Sentiment analysis for product reviews
   - Topic classification
   - Spam detection
   - Intent classification for chatbots

4. **Benchmark Different Models**:
   - Compare CNN vs RNN/LSTM
   - Try BERT or other transformers
   - Ensemble multiple models

## Learning Resources

### Understanding CNNs for Text
- Original Paper: [Kim (2014) - Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
- Blog Post: [Understanding CNNs for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)

### TensorFlow/Keras
- [Official TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Keras Documentation](https://keras.io/)

### Text Classification
- [A Comprehensive Guide to Text Classification](https://www.analyticsvidhya.com/blog/2020/12/a-comprehensive-guide-to-text-classification-in-nlp/)

## Getting Help

If you run into issues:

1. **Check the notebook**: Most common issues are addressed in the notebook cells
2. **Review error messages**: They often contain helpful information
3. **Search Stack Overflow**: Many TensorFlow/Keras issues are already solved
4. **Open an issue**: If you find a bug, open an issue on GitHub
5. **Read the documentation**: Check TensorFlow and Keras official docs

## Contributing

Found a bug or have an improvement? Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Ready to start learning?** Open the notebook and begin your journey into CNN-based text classification!

```bash
jupyter notebook cnn_text_classification.ipynb
```

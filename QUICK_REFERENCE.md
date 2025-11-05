# CNN Text Classification - Quick Reference

## Installation Commands

```bash
# Clone repository
git clone https://github.com/jidhu-mohan/CNN-Text-Classification.git
cd CNN-Text-Classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook cnn_text_classification.ipynb
```

## Key Hyperparameters

| Parameter | Default | Purpose | Typical Range |
|-----------|---------|---------|---------------|
| `MAX_FEATURES` | 10000 | Vocabulary size | 5000-20000 |
| `MAX_LEN` | 500 | Max sequence length | 200-500 |
| `EMBEDDING_DIM` | 128 | Embedding dimensions | 50-300 |
| `NUM_FILTERS` | 128 | Filters per size | 64-256 |
| `FILTER_SIZES` | [3,4,5] | N-gram sizes | [2,3,4], [3,4,5] |
| `DROPOUT_RATE` | 0.5 | Dropout probability | 0.3-0.6 |
| `BATCH_SIZE` | 128 | Training batch size | 32-256 |
| `EPOCHS` | 20 | Training epochs | 10-50 |

## Model Architecture Quick Reference

### Simple CNN
```python
Embedding(10000, 128) -> Dropout(0.5) -> Conv1D(128, 3, relu)
-> GlobalMaxPool -> Dense(128, relu) -> Dropout(0.5) -> Dense(1, sigmoid)
```

### Multi-Filter CNN (Kim 2014)
```python
Embedding(10000, 128) -> Dropout(0.5)
├─ Conv1D(128, 3, relu) -> GlobalMaxPool ─┐
├─ Conv1D(128, 4, relu) -> GlobalMaxPool ─┼─ Concatenate
└─ Conv1D(128, 5, relu) -> GlobalMaxPool ─┘
-> Dense(128, relu) -> Dropout(0.5) -> Dense(1, sigmoid)
```

## Common Code Snippets

### Build Model
```python
model = build_multi_filter_cnn(
    vocab_size=MAX_FEATURES,
    embedding_dim=EMBEDDING_DIM,
    max_length=MAX_LEN,
    num_filters=NUM_FILTERS,
    filter_sizes=FILTER_SIZES,
    dropout_rate=DROPOUT_RATE
)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

### Train Model
```python
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, model_checkpoint]
)
```

### Make Predictions
```python
# Single prediction
sentiment, confidence, score = predict_sentiment(
    "Your text here", model, word_index, MAX_LEN
)

# Batch predictions
predictions = model.predict(X_test)
y_pred = (predictions > 0.5).astype(int)
```

### Evaluate Model
```python
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(classification_report(y_test, y_pred))
```

### Save/Load Model
```python
# Save
model.save('model_name.keras')

# Load
loaded_model = tf.keras.models.load_model('model_name.keras')
```

## Preprocessing Pipeline

```python
# 1. Tokenization
tokenizer = Tokenizer(num_words=MAX_FEATURES)
tokenizer.fit_on_texts(texts)

# 2. Sequences
sequences = tokenizer.texts_to_sequences(texts)

# 3. Padding
X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')

# 4. Train/Val/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)
```

## Performance Tips

### Speed Up Training
- Increase batch size (if GPU memory allows)
- Reduce MAX_LEN for shorter sequences
- Use fewer epochs with early stopping
- Reduce vocabulary size (MAX_FEATURES)

### Improve Accuracy
- Use pre-trained embeddings (GloVe, Word2Vec)
- Increase model complexity (more filters)
- Try different filter size combinations
- Add more training data
- Implement data augmentation
- Use ensemble methods

### Reduce Overfitting
- Increase dropout rate
- Add L2 regularization
- Use early stopping
- Reduce model complexity
- Get more training data

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce batch_size, MAX_LEN, or NUM_FILTERS |
| Slow training | Increase batch_size, use GPU, reduce MAX_LEN |
| Poor accuracy | Increase epochs, try different hyperparameters |
| Overfitting | Increase dropout, use early stopping |
| Underfitting | Increase model capacity, train longer |

## Evaluation Metrics

```python
# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Precision, Recall, F1
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Classification Report
report = classification_report(y_test, y_pred)
```

## Keyboard Shortcuts (Jupyter)

| Shortcut | Action |
|----------|--------|
| `Shift + Enter` | Run cell and move to next |
| `Ctrl + Enter` | Run cell in place |
| `A` | Insert cell above (command mode) |
| `B` | Insert cell below (command mode) |
| `DD` | Delete cell (command mode) |
| `M` | Change to Markdown (command mode) |
| `Y` | Change to Code (command mode) |
| `Esc` | Enter command mode |
| `Enter` | Enter edit mode |

## Dataset Information

### IMDB Reviews
- Training: 25,000 reviews
- Testing: 25,000 reviews
- Classes: Binary (Positive/Negative)
- Auto-downloaded via `tensorflow.keras.datasets.imdb`

### Custom Dataset Format
```python
texts = ["review 1", "review 2", ...]  # List of strings
labels = [1, 0, 1, ...]                # List of labels (0 or 1)
```

## Expected Results

### IMDB Dataset Performance
- Simple CNN: ~86-87% accuracy
- Multi-Filter CNN: ~87-89% accuracy
- Training time: ~5-15 minutes (GPU) or ~30-60 minutes (CPU)

## Resources

- **Paper**: [Kim (2014) CNN for Sentence Classification](https://arxiv.org/abs/1408.5882)
- **Tutorial**: [Understanding CNNs for NLP](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)
- **TensorFlow Docs**: [Text Classification Guide](https://www.tensorflow.org/tutorials/keras/text_classification)

## Quick Experiments

### Experiment 1: Different Filter Sizes
```python
# Try different combinations
FILTER_SIZES = [2, 3, 4]     # Shorter n-grams
FILTER_SIZES = [3, 4, 5, 6]  # Include longer n-grams
FILTER_SIZES = [3]           # Single filter size
```

### Experiment 2: Embedding Dimensions
```python
EMBEDDING_DIM = 64   # Smaller embeddings
EMBEDDING_DIM = 256  # Larger embeddings
```

### Experiment 3: Model Depth
```python
# Add more conv layers
conv1 = Conv1D(128, 3, activation='relu')(embedding)
conv2 = Conv1D(128, 3, activation='relu')(conv1)
pool = GlobalMaxPooling1D()(conv2)
```

## Citation

```bibtex
@misc{cnn_text_classification,
  title = {CNN Text Classification: A Learning Project},
  year = {2025},
  url = {https://github.com/yourusername/CNN-Text-Classification}
}
```

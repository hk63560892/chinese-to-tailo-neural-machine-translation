# Neural Machine Translation with Enhanced Transformer Model

## Data Source
https://www.kaggle.com/competitions/machine-learning-2023nycu-translation/data

## Extention Model
http://tts001.iptcloud.net:8804/

## Introduction
This project presents a state-of-the-art approach to machine translation, leveraging an enhanced Transformer model. The model is designed to translate between two languages, incorporating advanced neural network techniques for improved accuracy and efficiency.

## Why Transformer?
The Transformer architecture, introduced by Vaswani et al., has revolutionized the field of machine translation. Its ability to handle sequences of data without the need for recurrent layers makes it highly efficient and effective for translation tasks. The Transformer's attention mechanism allows it to focus on different parts of the input sequence, providing a more nuanced understanding of context and meaning.

## Enhancements to the Standard Transformer
### LSTM and Convolutional Layers
- **LSTM Layers**: Long Short-Term Memory (LSTM) units are added to capture long-range dependencies and contextual information in the data, which are crucial for understanding and generating coherent translations.
- **Convolutional Layers**: Convolutional layers are used for feature extraction. They process the input data in a way that helps to identify patterns and structures, which is particularly useful for processing languages with complex morphologies.

### Positional Encoding
- Positional encoding is implemented to give the model a sense of word order, which is essential in translation for understanding grammatical and syntactical structures.

### Label Smoothing
- Label smoothing is a technique used to regularize the model. It makes the model's predictions less confident, which can prevent overfitting and lead to better generalization on unseen data.

## Data Processing and Vocabulary Building
- The data processing pipeline involves tokenizing the text data, building vocabularies for the source and target languages, and preparing the data for training.
- Custom tokenization functions are used for different languages, ensuring that the linguistic nuances of each language are appropriately handled.
- Uitilize the extentive model from NTUT to enhance the performace

## Model Architecture
The `TransformerModel` class encapsulates the enhanced Transformer architecture. Key components include:
- Embedding layers for source and target languages.
- Positional encoding for word order awareness.
- Convolutional and LSTM layers for advanced feature extraction and sequence modeling.
- A Transformer layer for the core translation mechanism.
- Layer normalization and dropout for regularization.

## Training and Evaluation
- The model is trained using a custom loop, with label smoothing loss and gradient clipping to ensure stable and effective learning.
- The training process is monitored using a progress bar, and loss metrics are logged for each epoch.

## Conclusion
This project demonstrates the effectiveness of combining Transformer architecture with LSTM and convolutional layers for machine translation. The use of label smoothing and positional encoding further enhances the model's performance, making it a robust solution for language translation tasks.


# Practical 1

## 1.1 Artificial Neural Network

```
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


data = pd.read_csv('amazon_cells_labelled.txt', sep='\t', names=['review', 'label']) 

# checking how many pos and neg reviews
review_counts = data['label'].value_counts()
print(f'Count of reviews by sentiment: {review_counts}')

nltk.download('punkt')
data['tokens'] = data['review'].apply(word_tokenize)

# create a vocabulary and map tokens to indices
vocab = {word: idx for idx, word in enumerate(set(word for review in data['tokens'] for word in review), 1)}
data['indexed'] = data['tokens'].apply(lambda x: [vocab[word] for word in x])

# pad sequences to a maximum length
max_len = max(len(review) for review in data['indexed'])
data['padded'] = data['indexed'].apply(lambda x: x + [0]*(max_len - len(x)))

# convert to tensord and split dataset
features = torch.tensor(data['padded'].tolist())
labels = torch.tensor(data['label'].tolist())

train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.2, random_state=88)
train_data = TensorDataset(train_x, train_y)
test_data = TensorDataset(test_x, test_y)
train_loader = DataLoader(train_data, batch_size=10)
test_loader = DataLoader(test_data, batch_size=10)

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embedded = self.embedding(x).mean(dim=1)
        hidden = self.relu(self.fc1(embedded))
        output = self.fc2(hidden)
        return output

model = TextClassifier(len(vocab)+1, 50, 100, 2)

criterion = nn.CrossEntropyLoss()
optimzier = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    for texts, labels in train_loader:
        optimzier.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimzier.step()
    print(f'Epoch {epoch+1}: Training Loss = {loss.item()}')

correct = 0
total = 0
predicted = []
actual = []
with torch.no_grad():
    for texts, labels in test_loader:
        outputs = model(texts)
        _, preds = torch.max(outputs.data, 1)  
        predicted.extend(preds.tolist())
        actual.extend(labels.tolist())

print('Test Accuracy:', accuracy_score(actual, predicted)*100)
print('Classification Report:\n', classification_report(actual, predicted))
print('Confusion Matrix:\n', confusion_matrix(actual, predicted))

def chatbot_respone(text):
    tokens = word_tokenize(text)
    indexed = [vocab.get(word, 0) for word in tokens]
    padded = indexed + [0]*(max_len - len(indexed))
    input_tensor = torch.tensor([padded])
    output = model(input_tensor)
    _, predicted = torch.max(output.data, 1)
    return "Positive" if predicted.item() == 1 else "Negative"

while True:
    user_input = input('Enter review: ')
    if user_input.lower() == 'quit':
        break
    response = chatbot_respone(user_input)
    print(response)
    
```

## 1.2 Transformers

```
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from torch.nn import functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
import math
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

nltk.download('punkt')

data = pd.read_csv('amazon_cells_labelled.txt', sep='\t', names=['review', 'label']) 

# tokenization
data['tokens'] = data['review'].apply(word_tokenize)

# creating vocabulary
word_counts = Counter(word for tokens in data['tokens'] for word in tokens)
vocab = {word: i + 1 for i, (word, _) in enumerate(word_counts.most_common())}
vocab['<pad>'] = 0

# encoding and padding
max_len = max(map(len, data['tokens']))
encoded_reviews = [[vocab[word] for word in tokens] + [0] * (max_len - len(tokens)) for tokens in data['tokens']]
labels = data['label'].values

# Splitting the dataset
train_x, test_x, train_y, test_y = train_test_split(encoded_reviews, labels, test_size=0.2, random_state=88)

# Dataloaders
class ReviewDataset(Dataset):
    def __init__(self, reviews, labels):
        self.reviews = reviews
        self.labels = labels

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        return torch.tensor(self.reviews[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


train_dataset = ReviewDataset(train_x, train_y)
test_dataset = ReviewDataset(test_x, test_y)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, ntoken, d_model, nhead, nhid, nlayers, dropout):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1) 
        output = self.decoder(output)
        return output.squeeze()


def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()  
    total_loss = 0
    correct = 0
    total = 0
    for text, labels in train_loader:
        text, labels = text.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output, labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predicted = (torch.sigmoid(output) > 0.5).int()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    average_loss = total_loss / len(train_loader)
    accuracy = correct / total

    if epoch >= 5:
        for param in model.parameters():
            if isinstance(param, nn.Dropout):
                param.p = 0.5

    return average_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for text, labels in test_loader:
            text, labels = text.to(device), labels.to(device)
            output = model(text)
            loss = criterion(output, labels.float())
            total_loss += loss.item()
            predicted = (torch.sigmoid(output) > 0.5).int()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    average_loss = total_loss / len(test_loader)
    accuracy = correct / total
    return average_loss, accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ntokens = len(vocab)  # size of vocabulary
d_model = 32  # embedding dimension
nhead = 8  # number of heads in the multiheadattention models
nhid = 256  # dimension of the feedforward network model
nlayers = 2  # number of nn.TransformerEncoderLayer
dropout = 0.3  # dropout rate

model = TransformerModel(ntokens, d_model, nhead, nhid, nlayers, dropout).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.85, 0.9), eps=1e-8, weight_decay=1e-3)

num_epochs = 100

for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, epoch)
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)
    print(
        f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

model.eval()
all_predictions = []
all_labels = []
with torch.no_grad():
    for text, labels in test_loader:
        text, labels = text.to(device), labels.to(device)
        output = model(text)
        predictions = (torch.sigmoid(output) > 0.5).int()
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
print('Test Accuracy:', accuracy_score(all_labels, all_predictions)*100)
print('Classification Report:')
print(classification_report(all_labels, all_predictions))
print('Confusion Matrix:')
print(confusion_matrix(all_labels, all_predictions))


def chatbot():
    model.eval() 
    print("Chatbot activated. Type 'quit' to exit.")

    while True:
        review = input("Enter a review: ")
        if review.lower() == 'quit':
            print("Chatbot deactivated.")
            break

        # tokenize and encode the review using the vocabulary
        tokens = [vocab.get(word, 0) for word in word_tokenize(review)]
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(device) 

        with torch.no_grad():
            output = model(tokens_tensor)
            prediction = torch.sigmoid(output).item()

        print("Positive" if prediction > 0.5 else "Negative")

chatbot()

```

## 1.3 Comparison

1. Compare the performance of the two models and explain in which scenarios you would prefer one over the other.


    ### ANN
    ```
    Classification Report:
    precision recall f1-score support

              0       0.79      0.81      0.80       100
              1       0.80      0.78      0.79       100

        accuracy                           0.80       200
       macro avg      0.80      0.80       0.79       200
    weighted avg      0.80      0.80       0.79       200

    Confusion Matrix:
    [[81 19]
    [22 78]]
    Test Accuracy: 79.5%
    Precision for class 0 (negative sentiment): 0.79
    Precision for class 1 (positive sentiment): 0.80
    Recall for class 0: 0.81
    Recall for class 1: 0.78


    25K Large 
    Classification Report:
                precision    recall  f1-score   support

            0       0.76      0.74      0.75      1960
            1       0.84      0.85      0.84      3040

        accuracy                           0.81      5000
       macro avg       0.80      0.80      0.80      5000
    weighted avg       0.81      0.81      0.81      5000

    Confusion Matrix:
    [[1458  502]
    [ 456 2584]]

    Test Accuracy: 80.84%
    Precision for class 0 (negative sentiment): 0.76
    Precision for class 1 (positive sentiment): 0.84
    Recall for class 0: 0.74
    Recall for class 1: 0.85
    ```

    

    - Best suited for relatively small datasets
    - Ideal for tasks with straightforward relationships between input features and output
    - Efficient and quick to train on smaller datasets
    - Performs well for data with linear or simple non-linear relationships


    ### Transformer
    ```
    Classification Report:
    precision recall f1-score support

              0       0.71      0.77      0.74       100
              1       0.75      0.68      0.71       100

        accuracy                           0.73       200

       macro avg      0.73      0.73       0.72       200
    weighted avg      0.73      0.7        0.72       200

    Confusion Matrix:
    [[77 23]
    [32 68]]
    Test Accuracy: 72.5%
    Precision for class 0 (negative sentiment): 0.71
    Precision for class 1 (positive sentiment): 0.75
    Recall for class 0: 0.77
    Recall for class 1: 0.68


    25K Large 
    Classification Report:
                precision    recall  f1-score   support

            0       0.83      0.80      0.81      1960
            1       0.87      0.89      0.88      3040

        accuracy                           0.86      5000
       macro avg       0.85      0.85      0.85      5000
    weighted avg       0.86      0.86      0.86      5000

    Confusion Matrix:
    [[1568  392]
    [ 323 2717]]

    Test Accuracy: 85.7%
    Precision for class 0 (negative sentiment): 0.83
    Precision for class 1 (positive sentiment): 0.87
    Recall for class 0: 0.80
    Recall for class 1: 0.89
    ```

    - Optimal for sequential data such as text or time series
    - Designed for tasks requiring the understanding of complex relationships and contextual information between input and output
    - Excels with large datasets due to the self-attention mechanism, enabling efficient capture of long-range dependencies
    - Demands higher computational resources for training compared to a simple ANN, owing to its architectural complexity



2. How did the two models’ complexity, accuracy, and efficiency differ? Did one model outperform the other in specific scenarios or tasks? If so, why?

    Model Complexity
    - ANN: Simpler model, faster training and lower computational requirements
    - Transformer: More complex model designed for sequential data processing, employing self-attention to capture contextual and long-range dependencies

    Accuracy
    - ANN: Achieved a test accuracy of 79.5%, outperforming the Transformer in this scenario
    - Transformer: Achieved a lower test accuracy of 72.5%

    Efficiency
    - ANN: Trains and evaluates quickly due to its simpler architecture
    - Transformer: Requires more training time and computational power because of its intricate self-attention mechanism

    Performance in Specific Scenarios
    - ANN: Outperformed the Transformer in terms of both accuracy and recall, indicating its suitability for datasets with simpler relationships between input features and outputs
    - Transformer: While it underperformed overall, its architecture might make it more suitable for tasks involving significant sequential or contextual relationships in the data

    Why the Difference? 
    - The given dataset lack complex sequential or contextual relationships, where a Transformer would shine
    - The task does not require long-range dependencies or self-attention mechanisms, aligning more with the strengths of the ANN.

  3. What insights did you obtain concerning data amount to train? Embed-diutilizedised? Architectural choices made?
      
       Embeddings, while critical for Transformers in processing sequential or complex data, may not add value for simpler tasks where features have direct relationships with the output. This underscores the need for thoughtful architectural selection—while Transformers are powerful for tasks involving sequential or contextual dependencies, their complexity can lead to overfitting or inefficiency in scenarios with limited data or simple patterns.

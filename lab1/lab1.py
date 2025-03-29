import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk
from data_loading_code import preprocess_pandas

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Get the outputs from the last time step
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out
    
def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # Reshape for LSTM: [batch_size, sequence_length, input_size]
            batch_x = batch_x.unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                batch_x = batch_x.unsqueeze(1)
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {total_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Accuracy: {(correct/total)*100:.2f}%')

# Chatbot prediction function
def predict_sentiment(model, vectorizer, text, device):
    model.eval()
    # Preprocess the input text
    text = text.lower()
    text = ''.join([c for c in text if c.isalnum() or c.isspace()])
    word_tokens = word_tokenize(text)
    filtered_sent = [w for w in word_tokens if not w in stopwords.words('english')]
    processed_text = " ".join(filtered_sent)
    
    # Vectorize
    vector = vectorizer.transform([processed_text]).todense()
    tensor = torch.from_numpy(np.array(vector)).type(torch.FloatTensor)
    tensor = tensor.unsqueeze(1).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        _, predicted = torch.max(output.data, 1)
    
    return predicted.item()

if __name__ == "__main__":
    # Load and preprocess data
    data = pd.read_csv("amazon_cells_labelled.txt", delimiter='\t', header=None)
    data.columns = ['Sentence', 'Class']
    data['index'] = data.index
    columns = ['index', 'Class', 'Sentence']
    data = preprocess_pandas(data, columns)

    # Split data
    training_data, validation_data, training_labels, validation_labels = train_test_split(
        data['Sentence'].values.astype('U'),
        data['Class'].values.astype('int32'),
        test_size=0.10,
        random_state=0,
        shuffle=True
    )

    # Vectorize data
    word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=50000, max_df=0.5, use_idf=True, norm='l2')
    training_data = word_vectorizer.fit_transform(training_data).todense()
    validation_data = word_vectorizer.transform(validation_data).todense()

    # Convert to tensors
    train_x_tensor = torch.from_numpy(np.array(training_data)).type(torch.FloatTensor)
    train_y_tensor = torch.from_numpy(np.array(training_labels)).long()
    val_x_tensor = torch.from_numpy(np.array(validation_data)).type(torch.FloatTensor)
    val_y_tensor = torch.from_numpy(np.array(validation_labels)).long()

    # Create data loaders
    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    val_dataset = TensorDataset(val_x_tensor, val_y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Model parameters
    input_size = train_x_tensor.shape[1]  # Size of TF-IDF vectors
    hidden_size = 128
    num_layers = 2
    num_classes = 2  # Positive or negative sentiment
    num_epochs = 10

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)

    # Train the model
    print("Training the model...")
    train_model(model, train_loader, val_loader, num_epochs, device)

    # Save the model
    torch.save(model.state_dict(), 'sentiment_lstm.pth')

    # Chatbot interface
    print("\nWelcome to the Sentiment Analysis Chatbot!")
    print("Type 'quit' to exit")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        prediction = predict_sentiment(model, word_vectorizer, user_input, device)
        response = "Positive sentiment! :)" if prediction == 1 else "Negative sentiment :("
        print(f"Bot: {response}")
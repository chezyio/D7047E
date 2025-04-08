import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk
import random

# Download required NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the ANN model
class SentimentANN(nn.Module):
    def __init__(self, input_size):
        super(SentimentANN, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 2)  # 2 outputs for positive/negative
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

# Preprocessing function
def preprocess_text(text, vectorizer=None):
    text = text.lower()
    text = ''.join([c for c in text if c.isalnum() or c.isspace()])
    word_tokens = word_tokenize(text)
    filtered_sent = [w for w in word_tokens if not w in stopwords.words('english')]
    processed_text = " ".join(filtered_sent)
    
    if vectorizer:
        transformed = vectorizer.transform([processed_text]).todense()
        return torch.from_numpy(np.array(transformed)).type(torch.FloatTensor)
    return processed_text

# Evaluation function
def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return accuracy, precision, recall, f1

# Train the model
def train_model():
    data = pd.read_csv("amazon_cells_labelled.txt", delimiter='\t', header=None)
    data.columns = ['Sentence', 'Class']
    
    vectorizer = TfidfVectorizer(max_features=5000, max_df=0.5, use_idf=True, norm='l2')
    X = vectorizer.fit_transform(data['Sentence'].values.astype('U')).todense()
    y = data['Class'].values.astype('int32')
    
    # Split into train, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    # This gives us 60% train, 20% validation, 20% test
    
    # Convert to tensors
    train_x_tensor = torch.from_numpy(np.array(X_train)).type(torch.FloatTensor)
    train_y_tensor = torch.from_numpy(np.array(y_train)).long()
    val_x_tensor = torch.from_numpy(np.array(X_val)).type(torch.FloatTensor)
    val_y_tensor = torch.from_numpy(np.array(y_val)).long()
    test_x_tensor = torch.from_numpy(np.array(X_test)).type(torch.FloatTensor)
    test_y_tensor = torch.from_numpy(np.array(y_test)).long()
    
    # Create data loaders
    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    val_dataset = TensorDataset(val_x_tensor, val_y_tensor)
    test_dataset = TensorDataset(test_x_tensor, test_y_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize model
    model = SentimentANN(train_x_tensor.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop with validation
    num_epochs = 100  # Reduced epochs for demonstration
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load('best_model.pt', weights_only=True))
    test_acc, test_prec, test_rec, test_f1 = evaluate_model(model, test_loader)
    
    print("\nTest Set Metrics:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall: {test_rec:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
    
    return model, vectorizer

class SentimentChatbot:
    def __init__(self):
        self.model, self.vectorizer = train_model()
        self.model = self.model.to(device)
        self.prompts = [
            "What do you think about your latest purchase?",
            "How would you rate your recent shopping experience?",
            "Tell me about a product you recently used.",
            "What’s your opinion on the last item you bought?"
        ]
        self.positive_responses = [
            "Glad you liked it!",
            "That's great to hear!",
            "Awesome, thanks for sharing!",
            "Sounds like a winner!"
        ]
        self.negative_responses = [
            "Sorry to hear that!",
            "That’s disconcerting.",
            "Thanks for the feedback.",
            "Ouch, that’s not good."
        ]
    
    def get_response(self, user_input):
        processed_input = preprocess_text(user_input, self.vectorizer)
        processed_input = processed_input.to(device)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(processed_input)
            sentiment = torch.argmax(prediction, dim=1).item()
        
        if sentiment == 1:
            return random.choice(self.positive_responses)
        else:
            return random.choice(self.negative_responses)
    
    def chat(self):
        print("Hello! I'm here to chat about your experiences.")
        while True:
            prompt = random.choice(self.prompts)
            print(f"\nBot: {prompt}")
            user_input = input("You: ")
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Bot: Goodbye!")
                break
                
            response = self.get_response(user_input)
            print(f"Bot: {response}")

# Run the chatbot
if __name__ == "__main__":
    chatbot = SentimentChatbot()
    chatbot.chat()
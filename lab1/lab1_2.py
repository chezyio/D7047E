import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk
import random

# Download required NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
vocab_size = 5000  # Will be updated after tokenization
block_size = 32    # Maximum context length
n_embd = 128       # Embedding dimension
n_head = 4         # Number of attention heads
n_layer = 2        # Number of transformer layers
dropout = 0.2
batch_size = 32
learning_rate = 3e-4
max_epochs = 20

# Preprocessing function from data_loading_code.py
def preprocess_pandas(data, columns):
    df_ = pd.DataFrame(columns=columns)
    data['Sentence'] = data['Sentence'].str.lower()
    data['Sentence'] = data['Sentence'].replace('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)
    data['Sentence'] = data['Sentence'].replace('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex=True)
    data['Sentence'] = data['Sentence'].str.replace('[^\w\s]','')
    data['Sentence'] = data['Sentence'].replace('\d', '', regex=True)
    
    processed_rows = []
    for index, row in data.iterrows():
        word_tokens = word_tokenize(row['Sentence'])
        filtered_sent = [w for w in word_tokens if not w in stopwords.words('english')]
        processed_rows.append({
            "index": row['index'],
            "Class": row['Class'],
            "Sentence": " ".join(filtered_sent[0:])
        })
    
    # Use concat instead of append
    df_ = pd.concat([df_, pd.DataFrame(processed_rows)], ignore_index=True)
    return data
# Transformer components from nano.py
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class SentimentTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.classification_head = nn.Linear(n_embd, 2)  # 2 classes: positive/negative

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        x = x.mean(dim=1)  # Average pooling over sequence length
        logits = self.classification_head(x)
        return logits

# Custom Dataset
class ReviewDataset(Dataset):
    def __init__(self, sentences, labels, stoi, max_len):
        self.sentences = sentences
        self.labels = labels
        self.stoi = stoi
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words = self.sentences[idx].split()
        encoded = [self.stoi.get(w, 0) for w in words[:self.max_len]]  # 0 for unknown words
        padded = encoded + [0] * (self.max_len - len(encoded)) if len(encoded) < self.max_len else encoded[:self.max_len]
        return torch.tensor(padded, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# Chatbot class
class SentimentChatbot:
    def __init__(self, model, stoi):
        self.model = model
        self.stoi = stoi
        self.prompts = [
            "What do you think about your latest purchase?",
            "How was your recent shopping experience?",
            "Tell me about a product you recently used.",
            "What's your opinion on your last purchase?"
        ]
        self.positive_responses = [
            "Great to hear that!",
            "Awesome, glad you liked it!",
            "Thanks for the positive feedback!",
            "Sounds like a good experience!"
        ]
        self.negative_responses = [
            "Sorry to hear that!",
            "That's too bad.",
            "Thanks for letting me know.",
            "I appreciate the honest feedback."
        ]

    def analyze_text(self, text):
        words = text.lower().split()
        encoded = [self.stoi.get(w, 0) for w in words[:block_size]]
        padded = encoded + [0] * (block_size - len(encoded)) if len(encoded) < block_size else encoded[:block_size]
        input_tensor = torch.tensor([padded], dtype=torch.long).to(device)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            prediction = torch.argmax(logits, dim=1).item()
        return prediction

    def chat(self):
        print("Hello! I'm here to chat about your experiences.")
        while True:
            prompt = random.choice(self.prompts)
            print(f"\nBot: {prompt}")
            user_input = input("You: ")
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Bot: Goodbye!")
                break
                
            sentiment = self.analyze_text(user_input)
            response = random.choice(self.positive_responses if sentiment == 1 else self.negative_responses)
            print(f"Bot: {response}")

# Training function with model saving
def train_model():
    global vocab_size  # Declare global at the start of the function
    
    # Load and preprocess data
    data = pd.read_csv("amazon_cells_labelled.txt", delimiter='\t', header=None)
    data.columns = ['Sentence', 'Class']
    data['index'] = data.index
    columns = ['index', 'Class', 'Sentence']
    data = preprocess_pandas(data, columns)

    # Create vocabulary
    all_words = ' '.join(data['Sentence']).split()
    unique_words = sorted(list(set(all_words)))
    stoi = {ch: i+1 for i, ch in enumerate(unique_words[:vocab_size-1])}  # +1 for padding (0)
    vocab_size = min(vocab_size, len(stoi) + 1)  # Update the global variable

    # Split data
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = ReviewDataset(train_data['Sentence'].values, train_data['Class'].values, stoi, block_size)
    test_dataset = ReviewDataset(test_data['Sentence'].values, test_data['Class'].values, stoi, block_size)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model
    model = SentimentTransformer(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)
        
        accuracy = correct / total
        print(f"Epoch {epoch+1}/{max_epochs}, Loss: {total_loss/len(train_loader):.4f}, Test Accuracy: {accuracy:.4f}")

    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'stoi': stoi
    }, 'sentiment_transformer.pt')
    print("Model saved to 'sentiment_transformer.pt'")

    return model, stoi

if __name__ == "__main__":
    # Train the model and save it
    model, stoi = train_model()
    
    # Create and run chatbot
    chatbot = SentimentChatbot(model, stoi)
    chatbot.chat()
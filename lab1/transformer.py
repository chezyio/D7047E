import torch
import torch.nn as nn
import math

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model):
        super(SingleHeadAttention, self).__init__()
        self.d_model = d_model
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, Q, K, V, mask=None):
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        return output, attention_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = SingleHeadAttention(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x, attn_weights

class Transformer(nn.Module):
    def __init__(self, d_model, num_layers, d_ff, max_len, vocab_size, dropout=0.1):
        super(Transformer, self).__init__()
        self.d_model = d_model  # Store d_model as an instance variable
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)  # Now this will work
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        attention_weights = []
        for layer in self.encoder_layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
            
        output = self.fc(x)
        return output, attention_weights

def main():
    d_model = 512    # embedding dimension
    num_layers = 6   # number of encoder layers
    d_ff = 2048      # feed-forward dimension
    max_len = 100    # maximum sequence length
    vocab_size = 10000  # vocabulary size
    batch_size = 32
    seq_len = 20

    # Create model
    model = Transformer(d_model, num_layers, d_ff, max_len, vocab_size)
    
    # Sample input (batch_size, seq_len)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    output, attention_weights = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of attention weight matrices: {len(attention_weights)}")
    print(f"Attention weights shape (per layer): {attention_weights[0].shape}")

if __name__ == "__main__":
    main()
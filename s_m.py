
import torch
import torch.nn as nn

# Improved LSTM Model
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=1, bidirectional=True):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)                          # (batch_size, seq_len, embed_dim)
        lstm_out, (hidden, _) = self.lstm(x)           # hidden: (num_layers * num_directions, batch, hidden_dim)

        # Concatenate final forward and backward hidden states
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (batch, hidden_dim * 2)
        else:
            hidden = hidden[-1]                                  # (batch, hidden_dim)

        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        return output


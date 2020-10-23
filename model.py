import torch
from torch import nn


class LSTM(nn.module):
    def __init__(self, dataset):
        super(LSTM,self).__init__()
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 3

        n_vocab = len(dataset.uniq_words)
        # embedding layer
        # Embedding layer converts word indexes to word vectors
        self.embedding = nn.Embedding(
            num_embeddings = n_vocab,
            embedding_dim = self.embedding_dim
        )
        self.lstm = nn.LSTM(
            input_size = self.lstm_size,
            hidden_size = self.lstm_size,
            num_layers = self.num_layers,
            dropout = 0.2

        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self,x,prev_state):
        embed = self.embedding()
        output, state = self.LSTM(embed, prev_state)
        logits = self.fc(output)

        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))

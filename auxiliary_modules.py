import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dim, rff, p_drop=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.activation_fn = nn.ReLU()
        self.dropout = nn.Dropout(p_drop)
        self.fc1 = nn.Linear(embedding_dim, embedding_dim * rff)
        self.fc2 = nn.Linear(embedding_dim * rff, embedding_dim)
        self.reset_parameter()

    def reset_parameter(self):
        # initialize linear layer right before relu: He initializer (kaiming normal)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)

        # initialize linear layer right before residual connection: zero initialize
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
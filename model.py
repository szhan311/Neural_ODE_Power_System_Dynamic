import torch
from torch import nn

class ODEBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ODEBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout4 = nn.Dropout(p=0.5)
        self.fc5 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, t, x):
        h = self.fc1(x)
        h = self.relu(h)
        # h = self.dropout1(h)
        h = self.fc2(h)
        h = self.relu(h)
        # h = self.dropout2(h)
        h = self.fc3(h)
        h = self.relu(h)
        # h = self.dropout3(h)
        h = self.fc4(h)
        h = self.relu(h)
        return self.fc5(h)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout4 = nn.Dropout(p=0.5)
        self.fc5 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self,t, x):
        h = self.fc1(x)
        h = self.relu(h)
        h = self.dropout1(h)
        h = self.fc2(h)
        h = self.relu(h)
        h = self.dropout2(h)
        h = self.fc3(h)
        h = self.relu(h)
        h = self.dropout3(h)
        h = self.fc4(h)
        h = self.relu(h)
        return self.fc5(h)


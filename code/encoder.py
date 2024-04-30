import torch
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx
from torch_geometric.transforms import RandomLinkSplit

class FeatureAutoencoder(torch.nn.Module):
    def __init__(self, in_channels, inner_hidden_channels, inner_out_channels, out_channels):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_channels, inner_hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(inner_hidden_channels, inner_hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(inner_hidden_channels, inner_out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(inner_out_channels, out_channels),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(out_channels, inner_out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(inner_out_channels, inner_hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(inner_hidden_channels, inner_hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(inner_hidden_channels, in_channels),
            torch.nn.ReLU()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class ModelLinear(torch.nn.Module):
    def __init__(self, number_input, hidden_one, hidden_two, number_output):
        super(ModelLinear, self).__init__()
        self.fc1 = torch.nn.Linear(number_input, hidden_one)
        self.ll2 = torch.nn.Linear(hidden_one, hidden_two)
        self.ll3 = torch.nn.Linear(hidden_two, number_output)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.ll2(x)
        x = self.ll3(x)
        return x

class ModelGraph(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        #x = self.sig(x)
        return x

class ModelRegression(torch.nn.Module):
    def __init__(self, number_input, number_hidden, number_output=1):
        super(ModelRegression, self).__init__()
        self.reg = torch.nn.Linear(number_input, number_hidden)
        self.reg2 = torch.nn.Linear(number_hidden, number_output)
        
    def forward(self, x):
        x = self.reg(torch.relu(x))
        x = self.reg2(x)
        return torch.relu(x)

# Ensemble Model for end-to-end training
class EnsembleModel(torch.nn.Module):
    def __init__(self, modelA, modelB, modelC):
        super(EnsembleModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.regression = modelC
        
    def forward(self, x1, x2, edge_index):
        x1 = self.modelA(x1)
        x2 = self.modelB(x2, edge_index)
        x = torch.cat((x1, x2), dim=1)
        x = self.regression(x)
        return x1, x2, x

def initialize_model(input_dimensions:int, linear_hidden_one:int, linear_hidden_two:int, linear_out:int, graph_hidden:int, graph_out:int, ensemble_hidden:int, ensemble_out:int):
    modelA = ModelLinear(number_input=input_dimensions, hidden_one=linear_hidden_one, hidden_two=linear_hidden_two, number_output=linear_out)
    modelB = ModelGraph(in_channels=input_dimensions, hidden_channels=graph_hidden, out_channels=graph_out)
    classifier = ModelRegression(number_input=linear_out+graph_out,number_hidden=ensemble_hidden, number_output=ensemble_out)
    model = EnsembleModel(modelA, modelB, classifier)
    return model


def train_model(model, G, X, Y):
    loss_fn = torch.nn.MSELoss()  # binary cross entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    n_epochs = 200
    losses = []
    edge_weights = torch.tensor([d['weight'] for u, v, d in G.edges(data=True)], dtype=torch.float).view(-1, 1)
    pyg_graph = from_networkx(G)
    pyg_graph.edge_attr = edge_weights
    transform = RandomLinkSplit(num_val = 0.0, num_test = 0.0, is_undirected=True, split_labels =True)
    train_data, val_data, test_data = transform(pyg_graph)
    optimizer.zero_grad() 
    model.zero_grad()
    for epoch in range(n_epochs):
        encoding, graph_encoding, y_pred = model(torch.stack(X),train_data.x.double(), train_data.edge_index)
        loss = loss_fn(y_pred, torch.stack(Y))
        #torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().item())
        if epoch%10 == 0:
            print("Epoch: "+str(epoch)+" Loss: "+str(losses[-1]))
            print("regression lost:" +str(loss_fn(y_pred[:, 0], torch.stack(Y)[:, 0])))
    # compute accuracy (no_grad is optional)
    with torch.no_grad():
        encoding, graph_encoding, y_pred = model(torch.stack(X),train_data.x.double(), train_data.edge_index)
        encoding = encoding.detach()
        graph_encoding = graph_encoding.detach()
        y_pred = y_pred.detach()

    print("Final  loss:"+str(losses[-1]))
    print("Final regression lost:" +str(loss_fn(y_pred[:, 0], torch.stack(Y)[:, 0])))
    #print(y_pred)
    return model

def create_latent_space(model, G, X):
    edge_weights = torch.tensor([d['weight'] for u, v, d in G.edges(data=True)], dtype=torch.float).view(-1, 1)
    pyg_graph = from_networkx(G)
    pyg_graph.edge_attr = edge_weights
    transform = RandomLinkSplit(num_val = 0.0, num_test = 0.0, is_undirected=True, split_labels =True)
    train_data, val_data, test_data = transform(pyg_graph)
    with torch.no_grad():
        encoding, graph_encoding, y_pred = model(torch.stack(X),train_data.x.double(), train_data.edge_index)
        encoding = encoding.detach()
        graph_encoding = graph_encoding.detach()
        y_pred = y_pred.detach()
    return encoding, graph_encoding, y_pred
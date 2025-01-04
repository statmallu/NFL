import numpy as np
from sklearn.metrics import roc_curve, auc, mean_squared_error
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import random
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

def update_edges_for_synthetic_nodes(augmented_features, similarity_threshold=0.8):
    """
    Updates edge index and weights for the graph with augmented nodes.
    
    Parameters:
    - augmented_features (torch.Tensor): Combined node features (shape: [num_nodes, num_features]).
    - similarity_threshold (float): Threshold for connecting nodes based on similarity.
    
    Returns:
    - edge_index (torch.Tensor): Updated edge indices.
    - edge_weights (torch.Tensor): Updated edge weights.
    """
    augmented_similarity_matrix = cosine_similarity(augmented_features.numpy())
    augmented_adjacency_matrix = (augmented_similarity_matrix > similarity_threshold).astype(int)
    augmented_G = nx.from_numpy_array(augmented_adjacency_matrix)

    # Extract edge index and weights
    edge_index = np.array(list(augmented_G.edges)).T
    edge_weights = np.array([augmented_similarity_matrix[i, j] for i, j in augmented_G.edges])

    # Convert to tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_weights = torch.tensor(edge_weights, dtype=torch.float32)

    return edge_index, edge_weights

def combine_original_and_synthetic_data(original_features, original_targets, synthetic_features, synthetic_targets):
    """
    Combines original and synthetic data for training.
    
    Parameters:
    - original_features (torch.Tensor): Original node features (shape: [num_nodes, num_features]).
    - original_targets (torch.Tensor): Original node targets (shape: [num_nodes, target_dim]).
    - synthetic_features (numpy.ndarray): Synthetic node features (shape: [num_synthetic_nodes, num_features]).
    - synthetic_targets (numpy.ndarray): Synthetic node targets (shape: [num_synthetic_nodes, target_dim]).
    
    Returns:
    - combined_features (torch.Tensor): Combined node features.
    - combined_targets (torch.Tensor): Combined node targets.
    """
    # Convert synthetic data to tensors
    synthetic_features = torch.tensor(synthetic_features, dtype=torch.float32)
    synthetic_targets = torch.tensor(synthetic_targets, dtype=torch.float32)

    # Combine original and synthetic data
    combined_features = torch.cat([original_features, synthetic_features], dim=0)
    combined_targets = torch.cat([original_targets, synthetic_targets], dim=0)

    return combined_features, combined_targets
    
def generate_synthetic_data_with_node_subsets(node_features, node_targets, num_synthetic_points=1000, subset_size=10, noise_std=0.05, seed=None):
    """
    Generates synthetic data by perturbing random subsets of nodes (variables).
    
    Parameters:
    - node_features (numpy.ndarray): Original node features (shape: [num_points, num_features]).
    - node_targets (numpy.ndarray): Original node targets (shape: [num_points, target_dim]).
    - num_synthetic_points (int): Total number of synthetic data points to generate.
    - subset_size (int): Number of nodes (variables) to include in each subset.
    - noise_std (float): Noise standard deviation for variability.
    - seed (int): Random seed for reproducibility.
    
    Returns:
    - synthetic_features (numpy.ndarray): Synthetic node features.
    - synthetic_targets (numpy.ndarray): Synthetic node targets.
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Ensure node_features and node_targets are NumPy arrays
    node_features = node_features.numpy() if isinstance(node_features, torch.Tensor) else node_features
    node_targets = node_targets.numpy() if isinstance(node_targets, torch.Tensor) else node_targets

    num_features = node_features.shape[1]
    synthetic_features = []
    synthetic_targets = []

    for _ in range(num_synthetic_points):
        # Randomly pick a base data point
        base_point_index = np.random.randint(0, node_features.shape[0])
        synthetic_point = node_features[base_point_index].copy()

        # Randomly select a subset of nodes
        selected_nodes = np.random.choice(range(num_features), size=subset_size, replace=False)

        # Perturb the selected nodes
        synthetic_point[selected_nodes] += np.random.normal(
            loc=0.0, scale=node_features.std(axis=0)[selected_nodes] * noise_std, size=subset_size
        )

        # Append the synthetic feature
        synthetic_features.append(synthetic_point)

        # Generate a synthetic target based on the base data point with noise
        synthetic_target = node_targets[base_point_index] + np.random.normal(
            loc=0.0, scale=node_targets.std(axis=0) * noise_std, size=node_targets.shape[1]
        )
        synthetic_targets.append(synthetic_target)

    return np.array(synthetic_features), np.array(synthetic_targets)




class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)  # First GCN layer
        self.conv2 = GCNConv(hidden_dim, hidden_dim)  # Second GCN layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)  # Fully connected layer 1
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Fully connected layer 2
        self.dropout = nn.Dropout(p=0.3)  # Dropout to reduce overfitting

    def forward(self, x, edge_index, edge_weight=None):
        # GCN Layers
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)  # Activation
        x = self.dropout(x)  # Dropout

        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)  # Activation
        x = self.dropout(x)  # Dropout

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
        
def load_graph_data(node_file, edge_file):
    # Load node data
    node_data = np.load(node_file)
    features = torch.tensor(node_data['features'], dtype=torch.float)  # Node features
    targets = torch.tensor(node_data['targets'], dtype=torch.float)  # Node targets

    # Load edge data
    edge_data = np.load(edge_file)
    
    # Combine sources and targets into a single NumPy array for efficient conversion
    edge_index_np = np.vstack([edge_data['sources'], edge_data['targets']])
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)  # Edge indices
    
    # Convert edge weights directly
    edge_weights = torch.tensor(edge_data['weights'], dtype=torch.float)  # Edge weights

    return features, targets, edge_index, edge_weights

    
       
        

def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_weight)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def validate(model, data, criterion, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_weight)
        loss = criterion(out[mask], data.y[mask])
    return loss.item()

def test(model, data):
    model.eval()
    with torch.no_grad():
        predictions = model(data.x, data.edge_index, data.edge_weight)
        return predictions[data.test_mask]


    
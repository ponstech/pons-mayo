"""
This script contains Graph Convolution Neural Network architectures.
"""
# utils/GN_models.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch.nn import LayerNorm
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import MessagePassing

########################################################################################
#################################### GCN Classifier ####################################
########################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNNClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout_p=0.5, drop_edge_p=0.2):
        """
        Graph Convolutional Neural Network (GCN) with edge weights for node classification.

        Parameters:
        - in_channels (int): Number of input features per node.
        - hidden_channels (int): Number of hidden units in the GCN layer.
        - num_classes (int): Number of output classes.
        - dropout_p (float): Dropout probability for node features.
        - drop_edge_p (float): Probability of dropping edges during training.
        """
        super(GCNNClassifier, self).__init__()

        # Graph convolution layers with edge weight support
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)

        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_channels, num_classes)

        # Regularization
        self.dropout = nn.Dropout(dropout_p)
        self.drop_edge_p = drop_edge_p  # Edge dropout probability

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass of GCN.

        Parameters:
        - x (Tensor): Node feature matrix (num_nodes, in_channels).
        - edge_index (Tensor): Graph connectivity (2, num_edges).
        - edge_weight (Tensor, optional): Edge weights (num_edges,).

        Returns:
        - Tensor: Output class logits (num_nodes, num_classes).
        """
        if self.training and edge_weight is not None:  
            # Apply edge dropout
            mask = torch.rand(edge_index.shape[1]) > self.drop_edge_p
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask] if edge_weight is not None else None

        # First GCN layer with edge weights
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Fully connected classification layer
        x = self.fc(x)

        return x  # Output class logits
    
########################################################################################
########################################################################################




########################################################################################
#################################### GCN with Dot Product ##############################
########################################################################################
class GCNN_Dot_Product(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        
        """
        This class defines a Graph Convolutional Neural Network (GCN) model with dot product for node classification tasks.
        The model incorporates node features into graph embeddings by element-wise multiplication.
        The final layer is a linear layer with softmax activation function to predict the class labels.
        
        Parameters:
        - in_channels (int): The number of input channels.
        - hidden_channels (int): The number of hidden channels.
        - num_classes (int): The number of classes for classification.
        """
        super(GCNN_Dot_Product, self).__init__()
        
        # First Graph Convolutional Layer
        self.conv1 = GCNConv(in_channels, in_channels)
        self.norm1 = nn.LayerNorm(in_channels) 

        # Fully Connected Layer for Classification
        self.fc = nn.Linear(in_channels, num_classes)  # Correct shape

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        
    def forward(self, x, edge_index):
        """
        Forward pass through the network.

        Parameters:
        - x (Tensor): Graph node feature matrix (num_nodes, in_channels).
        - edge_index (Tensor): Graph connectivity.

        Returns:
        - Tensor: Output class probabilities.
        """
        original_input = x.clone()  # Save the original input for the multipication connection
        
        # First GCN Layer
        x = self.conv1(x, edge_index)
        x = self.norm1(x) 
        x = F.relu(x)
        x = self.dropout(x)
 

        # Element-wise multiplication with input feature embeddings (NOT matrix multiplication)
        x = torch.mul(x, original_input)  # Shape (num_nodes, hidden_channels)
        
        # Fully connected classification layer
        x = self.fc(x)  # Shape (num_nodes, num_classes)

        return x  # Output class probabilities


########################################################################################
########################################################################################



########################################################################################
############################ GCN with Attention Concatenation ##########################
########################################################################################
class GCNN_Concat_Attention(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        
        """
        This class defines a Graph Convolutional Neural Network (GCN) model with attention concatenation for node classification tasks.
        The model combines graph embeddings with node features using an attention mechanism.
        The final layer is a linear layer with softmax activation function to predict the class labels.
        
        Parameters:
        - in_channels (int): The number of input channels.
        - hidden_channels (int): The number of hidden channels.
        - num_classes (int): The number of classes for classification.
        """
        super(GCNN_Concat_Attention, self).__init__()
        
        # First Graph Convolutional Layer
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels) 

        # Fully Connected Layer for Classification
        self.fc = nn.Linear(hidden_channels + in_channels, num_classes)  # Correct shape

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Attention weights
        self.attention_weights = nn.Linear(in_channels + hidden_channels, 1)

    def forward(self, x, edge_index):
        """
        Forward pass through the network.

        Parameters:
        - x (Tensor): Graph node feature matrix (num_nodes, in_channels).
        - edge_index (Tensor): Graph connectivity.

        Returns:
        - Tensor: Output class probabilities.
        """
        original_input = x.clone()  # Save the original input for the concatenation
        
        # First GCN Layer
        x = self.conv1(x, edge_index)
        x = self.norm1(x) 
        x = F.relu(x)
        x = self.dropout(x)
 

        # Concatenate with input feature embeddings
        combined_features = torch.cat((x, original_input), dim=1)  # Shape (num_nodes, hidden_channels + in_channels)
        
        # Compute attention scores
        attention_scores = torch.sigmoid(self.attention_weights(combined_features))
        x = combined_features * attention_scores  # Weighted combination

        # Fully connected classification layer
        x = self.fc(x)  # Shape (num_nodes, num_classes)

        return x # Output class probabilities
    
########################################################################################
########################################################################################


########################################################################################
##################### GCN with Dot Product and Residual Layer ##########################
########################################################################################
class GCNN_Prod_Res(nn.Module):
    """
    This class defines a Graph Convolutional Neural Network (GCN) model with a residual connection and dot product operation for node classification tasks.
    The model consists of a single GCN layer followed by layer normalization, ReLU activation, and dropout for regularization.
    The output of the GCN layer is then multiplied with the input feature embeddings using dot product, and a residual connection is added.
    The final layer is a linear layer with softmax activation function to predict the class labels.

    Parameters:
    - in_channels (int): The number of input channels.
    - hidden_channels (int): The number of hidden channels.
    - num_classes (int): The number of classes for classification.
    """

    def __init__(self, in_channels, hidden_channels, num_classes, dropout_p = 0.5):
        super(GCNN_Prod_Res, self).__init__()
        
        self.conv1 = GCNConv(in_channels, in_channels)  # Fix hidden_channels = in_channels
        self.norm1 = nn.LayerNorm(in_channels) 
        self.fc = nn.Linear(in_channels, num_classes)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, edge_index):
        """
        Forward pass through the network with residual connection and dot product operation.

        Parameters:
        - x (Tensor): Graph node feature matrix (num_nodes, in_channels).
        - edge_index (Tensor): Graph connectivity.
        
        Returns:
        - Tensor: Output class probabilities.
        """
        residual = x.clone()  # Save the original input for the residual connection

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Multiply with input feature embeddings
        x = torch.mul(x, residual)
        x = self.norm1(x) 
        
        # Add residual connection
        x += residual  # Add the original input back

        x = self.fc(x)
        return x
    
########################################################################################
########################################################################################




########################################################################################
############################ Graph Attention Network ###################################
########################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import LayerNorm

class GATClassifier(nn.Module):
    """
    Graph Attention Network (GAT) with edge weights for node classification.
    Edge weights represent confidence scores to adjust attention mechanisms.
    """
    def __init__(self, in_channels, hidden_channels, num_classes, heads=4, dropout_p=0.5):
        super(GATClassifier, self).__init__()
        
        # First GAT layer with attention heads
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True, edge_dim=1)
        self.norm1 = LayerNorm(hidden_channels * heads)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_channels * heads, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass through the Graph Attention Network.

        Parameters:
        - x (Tensor): Node feature matrix (num_nodes, in_channels).
        - edge_index (Tensor): Graph connectivity.
        - edge_weight (Tensor, optional): Edge weights (confidence scores of pseudo-labels).

        Returns:
        - Tensor: Class logits (num_nodes, num_classes).
        """

        # Apply GATConv with edge weights
        x = self.conv1(x, edge_index, edge_attr=edge_weight)

        # Normalize and apply activation
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Final classification layer
        x = self.fc(x)

        return x
########################################################################################
########################################################################################




########################################################################################
############################### GraphSAGE Network ######################################
########################################################################################
class GraphSAGEClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, num_classes=6, aggr='mean', normalize=True,  dropout_p = 0.5, drop_edge_p=0.2):
        """
        This class defines a GraphSAGE model for node classification tasks.
        The model consists of two GraphSAGE layers, each followed by layer normalization and ReLU activation functions.
        The final layer is a linear layer with softmax activation function to predict the class labels.
        
        Parameters:
        - in_channels (int): The number of input channels.
        - hidden_channels (int): The number of hidden channels.
        - num_classes (int): The number of classes for classification.
        - aggr (str): The aggregation method to use. Can be 'mean', 'add', 'max'.
        """
        super(GraphSAGEClassifier, self).__init__()
        self.conv1 = SAGEConv(in_channels, out_channels=hidden_channels, normalize=True, aggr=aggr, project=True)
        self.norm1 = LayerNorm(hidden_channels)
        self.fc = nn.Linear(hidden_channels, num_classes)
        self.dropout = nn.Dropout(dropout_p)
        self.drop_edge_p = drop_edge_p

    def forward(self, x, edge_index):  # No edge weights, due to random sampling of neighbors
        """
        Forward pass through the GraphSAGE Network.

        Parameters:
        - x (Tensor): Graph node feature matrix (num_nodes, in_channels).
        - edge_index (Tensor): Graph connectivity.

        Returns:
        - Tensor: Output class probabilities.
        """
        if self.training:  # Only drop edges during training
            mask = torch.rand(edge_index.shape[1]) > self.drop_edge_p
            edge_index = edge_index[:, mask]
            
        x = self.conv1(x, edge_index)  
        x = F.relu(x)
        x = self.dropout(x)


        x = self.fc(x)
        return x  # Output class probabilities
    
########################################################################################
########################################################################################




########################################################################################
############################### Message Passing GN #####################################
########################################################################################




class GNClassifier(MessagePassing):
    """
    Graph Neural Network (GN) using MessagePassing for node classification.
    
    Args:
        in_channels (int): Feature dimension of nodes.
        hidden_dim (int): Number of hidden units in the MLP layers.
        num_classes (int): Number of output classes (default = 6).
        message_dim (int): Dimension of the message (embedding size used in message passing).
        aggregation (str): Aggregation method ('add', 'mean', 'max').
    """
    def __init__(self, in_channels, hidden_channels, num_classes=6, aggregation='add', lambda_l1=1e-4):
        super(GNClassifier, self).__init__(aggr=aggregation)  # Defines how messages are aggregated

        self.lambda_l1 = lambda_l1
        
        # Edge model: Computes messages between connected nodes
        self.edge_model = nn.Sequential(
            nn.Linear(2*in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # Node model: Updates node features using aggregated messages
        self.node_model = nn.Sequential(
            nn.Linear(hidden_channels + in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, num_classes),  # Output dimension = number of classes (6)
        )

    def forward(self, x, edge_index, augmentation=False):
        """
        Forward pass: Propagate information across graph nodes with optional data augmentation.

        Args:
            x (Tensor): Node feature matrix (num_nodes, in_channels).
            edge_index (Tensor): Graph connectivity matrix (2, num_edges).
            augmentation (bool): Whether to apply data augmentation.

        Returns:
            Tensor: Class logits (num_nodes, num_classes).
        """
        if augmentation:
            # Apply Gaussian noise to node features for augmentation
            noise = torch.randn_like(x) * 0.01  # Small noise factor
            x = x + noise  # Add noise to node features
        
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        """
        Compute messages from source (x_j) to target (x_i).

        Args:
            x_i (Tensor): Features of target nodes.
            x_j (Tensor): Features of source nodes.

        Returns:
            Tensor: Computed messages.
        """
        # Compute L1 norm for regularization
        
        message = self.edge_model(torch.cat([x_i, x_j], dim=1))  # Compute message for edge
        
        self.l1_penalty = torch.norm(message, p=1)  # L1 norm of message vectors
        
        # Compute cosine similarity between message vectors
        cosine_sim = F.cosine_similarity(message, x_i, dim=1)
        self.cosine_penalty = torch.mean(1 - cosine_sim)  # Minimize 1 - similarity
        
        return message

    def update(self, aggr_out, x):
        """
        Update node features by combining input features with aggregated messages.

        Args:
            aggr_out (Tensor): Aggregated messages.
            x (Tensor): Node features.

        Returns:
            Tensor: Updated node embeddings.
        """
        return self.node_model(torch.cat([x, aggr_out], dim=1))  # Combine node features with messages

    def loss(self, graph, augmentation):
        """
        Compute classification loss.

        Args:
            graph (Data): Graph object containing node features, edge indices, and ground truth labels.

        Returns:
            Tensor: Cross-entropy loss.
        """
        
        # Apply cosine similarity penalty
        cosine_reg = self.cosine_penalty
        
        logits = self.forward(graph.x, graph.edge_index, augmentation)  # Get model predictions
    
        # ✅ Confidence-Weighted Cross-Entropy Loss
        ce_loss = F.cross_entropy(logits, graph.y, reduction='none')  # Compute per-node loss
        weighted_ce_loss = (graph.pseudo_probs * ce_loss).sum() / graph.pseudo_probs.sum()  # Scale loss by confidence scores

        # ✅ L1 Regularization on Message Vectors
        l1_loss = self.lambda_l1 * self.l1_penalty  # Scale by lambda

        # ✅ Compute Class Centroids in Feature Space
        class_centroids = []
        for class_id in torch.unique(graph.y):
            class_mask = (graph.y == class_id)
            class_centroid = torch.mean(self.forward(graph.x, graph.edge_index, augmentation)[class_mask], dim=0)  
            class_centroids.append(class_centroid)

        class_centroids = torch.stack(class_centroids)  # Shape: (num_classes, feature_dim)

        # ✅ Compute Confidence-Weighted Cosine Similarity Loss
        node_cosine_sim = F.cosine_similarity(
            self.forward(graph.x, graph.edge_index, augmentation), class_centroids[graph.y], dim=1
        )
        cosine_loss = torch.mean((1 - node_cosine_sim) * graph.pseudo_probs)  # Scale by confidence

        # ✅ Total Loss = Confidence-Weighted CE + Confidence-Weighted Cosine + L1 Reg
        total_loss = weighted_ce_loss + cosine_loss + l1_loss  

        return total_loss
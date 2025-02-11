 
import pandas as pd
import matplotlib.pyplot as plt
import os
import gzip
import numpy as np
import os
import mne
import json
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from torch.nn import Module, Linear
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
import networkx as nx
from collections import Counter

mne.set_log_level("ERROR")
  

# Commented out IPython magic to ensure Python compatibility.
os.chdir('DATASET')

gb_df = pd.read_csv("metadata_guineabissau.csv")

gb_df.head(5)

ni_df = pd.read_csv("metadata_nigeria.csv")

ni_df.head(5)

"""## Datasets transformations"""

def file_exists(file_path):
    return os.path.isfile(file_path)

"""### Nigeria

Adding new columns
"""

ni_df["Eyes.condition"] = ni_df["first_condition"].apply(lambda x: "open-3min-then-closed-2min" if x == "open" else "closed-3min-then-open-2min")
# ni_df["csv.file"] = ni_df["csv.file"].apply(lambda x: "EEGs_Nigeria/"+x)
ni_df["csv.file"] = ni_df["csv.file"].apply(lambda x: "EEGs_Nigeria/"+x)
ni_df["Group"] = ni_df["Group"].apply(lambda x: x.title())
ni_df["subject.id"] = ni_df["subject.id"].apply(lambda x: "NI-" + str(x))
ni_df["Country"] = "Nigeria"
ni_df['file_exists'] = ni_df['csv.file'].apply(lambda x: file_exists(x))
ni_df = ni_df.drop(columns=['session.id'])

def count_rows(file):
  data = pd.read_csv(file, compression='gzip',
                   on_bad_lines='skip')
  return len(data)

ni_df.head(5)

"""Some data cleaning will be required"""

ni_df[ni_df["recordedPeriod"]<200]

(ni_df[ni_df["recordedPeriod"]<200])["remarks"].unique()

samples_to_remove = ["NI-124", "NI-536", "NI-522", "NI-508", "NI-515", "NI-580"]

ni_df = ni_df[(ni_df['file_exists']) & (~ni_df['subject.id'].isin(samples_to_remove))]
ni_df = ni_df[(ni_df['file_exists']) & (~ni_df['subject.id'].isin(samples_to_remove))]

# Check the result
print(ni_df)

ni_df["file_rows_count"] = ni_df["csv.file"].apply(lambda x: count_rows(x))

ni_df.head(5)

"""### Guinea-Bissau

Adding new columns
"""

gb_df["first_condition"] = gb_df['Eyes.condition'].str.split('-').str[0]
gb_df["csv.file"] = gb_df["subject.id"].apply(lambda x: "EEGs_Guinea-Bissau/signal-" + str(x) + ".csv.gz")
gb_df["Group"] = gb_df["Group"].apply(lambda x: x.title())
gb_df["subject.id"] = gb_df["subject.id"].apply(lambda x: "GB-" + str(x))
gb_df["Country"] = "Guinea Bissau"
gb_df['file_exists'] = gb_df['csv.file'].apply(lambda x: file_exists(x))

gb_df = gb_df[(gb_df['file_exists'])]

gb_df["file_rows_count"] = gb_df["csv.file"].apply(lambda x: count_rows(x))

gb_df.head(5)

"""## Display data"""

def plot_frequency_all(path):

    data = pd.read_csv(path, compression='gzip',
                       on_bad_lines='skip')

    # Get the x-axis (first column) and the y-values (all other columns)
    x = data.iloc[:, 0]
    columns = data.columns[1:15]  # Skip the first column for the y-values

    # Create subplots
    fig, axes = plt.subplots(nrows=len(columns), ncols=1, figsize=(12, 20), sharex=True)

    for i, col in enumerate(columns):
        axes[i].plot(x, data[col], label=col)
        axes[i].set_title(col)
        axes[i].set_ylabel("Value")
        axes[i].grid(True)
        axes[i].legend()

    # Set common x-label
    plt.xlabel("Index")
    # plt.tight_layout()
    plt.show()

def plot_frequency_closed(path, ni_df):
 
    # Lookup row_count and first_condition in ni_df
    row_info = ni_df[ni_df['csv.file'] == path]

    if row_info.empty:
        raise ValueError(f"No matching entry found for file path: {path}")

    row_count = row_info['file_rows_count'].values[0]
    first_condition = row_info['first_condition'].values[0]

    # Calculate the number of rows to read (0.4 of the total rows)
    rows_to_read = int(row_count * 0.4)

    # Load the relevant portion of data
    with gzip.open(path, 'rt') as f:
        if first_condition == "closed":
            # Read the first 0.4 of rows, including the header
            data = pd.read_csv(f, nrows=rows_to_read + 1)  # +1 to include the header
        elif first_condition == "open":
            # Calculate how many rows to skip but include the header
            skip_rows = row_count - rows_to_read
            data = pd.read_csv(f, skiprows=range(1, skip_rows + 1))  # Skip initial rows but not the header
        else:
            raise ValueError(f"Invalid first_condition: {first_condition}. Must be 'open' or 'closed'.")

    # Get the x-axis (first column) and tcalculate_plv_matrixhe y-values (all other columns)
    x = data.iloc[:, 0]
    columns = data.columns[1:15]  # Use the first 14 columns after the x-axis for plotting

    # Create subplots
    fig, axes = plt.subplots(nrows=len(columns), ncols=1, figsize=(12, 20), sharex=True)

    for i, col in enumerate(columns):
        axes[i].plot(x, data[col], label=col)
        axes[i].set_title(col)
        axes[i].set_ylabel("Value")
        axes[i].grid(True)
        axes[i].legend()

    # Set common x-label
    plt.xlabel("Index")
    plt.tight_layout()
    plt.show()

"""Let's see some example data from samples"""

data = pd.read_csv('EEGs_Nigeria/signal-10-1.csv.gz',    on_bad_lines='skip')
data
data.info()
 
"""Same for Guinea-Bissau data."""

data = pd.read_csv('EEGs_Guinea-Bissau/signal-1.csv.gz', compression='gzip',
                   on_bad_lines='skip')
data
data.info()
 

def load_eeg_from_gz(file_path, row_count, first_condition):
    # Calculate the number of rows to read (0.4 of the total rows)
    rows_to_read = int(row_count * 0.4)

    with gzip.open(file_path, 'rt') as f:
      if first_condition == "closed":
            # Read the first 0.4 of rows
            eeg_data = pd.read_csv(f, nrows=rows_to_read)
      elif first_condition == "open":
            # Skip rows to get the last 0.4 of rows
            skip_rows = row_count - rows_to_read
            eeg_data = pd.read_csv(f, skiprows=range(1, skip_rows + 1))  # Skip the header and initial rows
      else:
            raise ValueError(f"Invalid first_condition: {first_condition}. Must be 'open' or 'closed'.")

    return eeg_data

 

# @title Default title text
def preprocess_eeg(raw_data, low_freq=1, high_freq=30, sfreq=128):
    # Extract columns 1–14 (EEG data)
    eeg_data = raw_data.iloc[:, 0:14]
    ch_names = eeg_data.columns.tolist()  # Channel names (columns 1–14)
    ch_types = ["eeg"] * len(ch_names)  # All channels as EEG

    # Create MNE RawArray object
    raw_array = mne.io.RawArray(
        eeg_data.values.T,
        mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    )
    raw_array .set_eeg_reference()

    # Bandpass filtering (1–30 Hz)
    raw_array.filter(low_freq, high_freq, fir_design='firwin')
 
    raw_array = normalize_amplitude(raw_array)
    return raw_array


def preprocess_eeg_with_epochs(raw_data, duration=5, overlap=1, sfreq=128):
   
    # Extract EEG data (columns 1-14)
    eeg_data = raw_data.iloc[:, 0:14]
    ch_names = eeg_data.columns.tolist()
    ch_types = ["eeg"] * len(ch_names)

    # Create MNE RawArray
    raw_array = mne.io.RawArray(
        eeg_data.values.T,
        mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    )

    # Bandpass filtering (1-30 Hz)
    raw_array.filter(1, 30, fir_design='firwin')

    # Create epochs
    epochs = mne.make_fixed_length_epochs(raw_array, duration=duration, overlap=overlap)

    # Drop bad epochs
    epochs = epochs.drop_bad()

    return epochs, len(epochs)

def expand_labels_for_epochs(original_labels, n_epochs_per_sample):
   
    expanded_labels = []
    for label, n_epochs in zip(original_labels, n_epochs_per_sample):
        expanded_labels.extend([label] * n_epochs)
    return np.array(expanded_labels)

def normalize_amplitude(raw_data, technique='min_max', axis=None, keepdims=True):
  
    # Get the data array
    if hasattr(raw_data, 'get_data'):
        raw_data_values = raw_data.get_data()
    else:
        raw_data_values = raw_data.copy()

    # Compute statistics along specified axis
    min_val = raw_data_values.min(axis=axis, keepdims=keepdims)
    max_val = raw_data_values.max(axis=axis, keepdims=keepdims)
    mean_val = raw_data_values.mean(axis=axis, keepdims=keepdims)
    std_val = raw_data_values.std(axis=axis, keepdims=keepdims)

    # Handle division by zero cases
    eps = 1e-8

    # Apply normalization based on selected technique
    if technique == 'min_max':
        denominator = (max_val - min_val)
        denominator = np.where(denominator == 0, eps, denominator)
        normalized_data = (raw_data_values - min_val) / denominator

    elif technique == 'min_max_symmetric':
        denominator = (max_val - min_val)
        denominator = np.where(denominator == 0, eps, denominator)
        normalized_data = 2 * ((raw_data_values - min_val) / denominator) - 1

    elif technique == 'mean_std':
        denominator = std_val
        denominator = np.where(denominator == 0, eps, denominator)
        normalized_data = (raw_data_values - mean_val) / denominator

    elif technique == 'mean_only':
        normalized_data = raw_data_values - mean_val

    else:
        raise ValueError(f"Unknown normalization technique: {technique}")

    # Return data in the same format as input
    if hasattr(raw_data, 'get_data'):
        raw_data._data = normalized_data
        return raw_data
    else:
        return normalized_data


def calculate_plv_matrix(eeg_data):
    n_channels = len(eeg_data.info['ch_names'])
    plv_matrix = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            signal1 = eeg_data.get_data(picks=i).flatten()
            signal2 = eeg_data.get_data(picks=j).flatten()
            plv_matrix[i, j] = calculate_plv(signal1, signal2)
    return plv_matrix

def calculate_plv(signal1, signal2):
    phase1 = np.angle(signal1)
    phase2 = np.angle(signal2)
    return np.abs(np.mean(np.exp(1j * (phase1 - phase2))))
 
def compute_features(eeg_data):
 
    features = {}
    sfreq = eeg_data.info['sfreq']

    for ch_name in eeg_data.info['ch_names']:
        signal = eeg_data.get_data(picks=ch_name).flatten()

        # Katz FD
        L = np.sum(np.sqrt(np.diff(signal)**2 + 1))
        d = np.max(signal) - np.min(signal)
        katz_fd = np.log10(L) / np.log10(d)

        # Band energy (delta, theta, alpha, beta)
        freqs, psd = mne.time_frequency.psd_array_welch(signal, sfreq=sfreq, n_fft=256)
        delta_energy = np.sum(psd[(freqs >= 1) & (freqs < 4)])
        theta_energy = np.sum(psd[(freqs >= 4) & (freqs < 8)])
        alpha_energy = np.sum(psd[(freqs >= 8) & (freqs < 13)])
        beta_energy = np.sum(psd[(freqs >= 13) & (freqs < 30)])

        features[ch_name] = [katz_fd, delta_energy, theta_energy, alpha_energy, beta_energy]

    return features
 

def create_graph_from_plv(plv_matrix, channel_names):
    """
    Create a graph from a PLV matrix using original channel names as node labels.
    """
    G = nx.Graph()

    # Add nodes with channel names as labels
    for i, name in enumerate(channel_names):
        G.add_node(name)  # Use the original channel name instead of numeric ID

    # Add edges with weights
    for i in range(len(plv_matrix)):
        for j in range(i + 1, len(plv_matrix)):
            if plv_matrix[i, j] > 0:  # Include edges with positive weights
                G.add_edge(channel_names[i], channel_names[j], weight=plv_matrix[i, j])

    return G

 

def process_all_files(metadata, output_dir, feature_output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(feature_output_dir, exist_ok=True)

    for _, row in metadata.iterrows():
        # Load EEG data from .gz file
        eeg_data = load_eeg_from_gz(row['csv.file'], row['file_rows_count'], row['first_condition'])

        # Filtering and processing the signal
        raw_array = preprocess_eeg(eeg_data.iloc[:, 1:15])

        # Calculate PLV
        plv_matrix = calculate_plv_matrix(raw_array)

        # Create the graph
        graph = create_graph_from_plv(plv_matrix, raw_array.info['ch_names'])

        # Save the graph
        nx.write_gml(graph, os.path.join(output_dir, f"{row['subject.id']}_graph.gml"))

        # Compute features and save separately
        features = compute_features(raw_array)
        feature_file = os.path.join(feature_output_dir, f"{row['subject.id']}_features.json")
        with open(feature_file, 'w') as f:
            json.dump(features, f)


def process_all_files_with_epochs(metadata, output_dir, feature_output_dir, duration=5, overlap=1):
 
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(feature_output_dir, exist_ok=True)

    n_epochs_per_sample = []

    for idx, row in metadata.iterrows():
        # Load EEG data
        eeg_data = load_eeg_from_gz(row['csv.file'], row['file_rows_count'], row['first_condition'])
        print('sample')
        print(idx)
        # Preprocess and create epochs
        epochs, n_epochs = preprocess_eeg_with_epochs(eeg_data, duration=duration, overlap=overlap)
        n_epochs_per_sample.append(n_epochs)

        # Process each epoch
        for epoch_idx in range(n_epochs):
            epoch_data = epochs[epoch_idx]
           
            # Calculate PLV for this epoch
            plv_matrix = calculate_plv_matrix(epoch_data)

            # Create graph for this epoch
            graph = create_graph_from_plv(plv_matrix, epoch_data.info['ch_names'])

            # Save graph with epoch index in filename
            nx.write_gml(graph, os.path.join(output_dir, f"{row['subject.id']}_epoch{epoch_idx}_graph.gml"))

            # Compute and save features for this epoch
            features = compute_features(epoch_data)
            feature_file = os.path.join(feature_output_dir, f"{row['subject.id']}_epoch{epoch_idx}_features.json")
            with open(feature_file, 'w') as f:
                json.dump(features, f)

    return n_epochs_per_sample
 

def load_graphs_and_labels_with_features(gml_dir, feature_dir, metadata):
    """
    Load .gml graphs, enrich with features, and convert to PyTorch Geometric format.
    """
    graphs = []
    labels = []

    label_encoder = LabelEncoder()
    metadata['encoded_group'] = label_encoder.fit_transform(metadata['Group'])

    for _, row in metadata.iterrows():
        gml_file = os.path.join(gml_dir, f"{row['subject.id']}_graph.gml")
        feature_file = os.path.join(feature_dir, f"{row['subject.id']}_features.json")

        if os.path.exists(gml_file) and os.path.exists(feature_file):
            # Load graph
            graph = nx.read_gml(gml_file)

            # Load features
            with open(feature_file, 'r') as f:
                features = json.load(f)

            # Map node labels to numeric indices
            node_mapping = {node: idx for idx, node in enumerate(graph.nodes)}

            # Convert edges to numeric indices
            edge_index = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in graph.edges], dtype=torch.long).t().contiguous()

            # Convert node features to tensor
            node_features = torch.tensor([features[node] for node in graph.nodes], dtype=torch.float32)

            # Get edge attributes (e.g., weights)
            edge_attr = torch.tensor([data['weight'] for _, _, data in graph.edges(data=True)], dtype=torch.float32)

            # Create PyTorch Geometric data object
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(graph.nodes), y=torch.tensor([row['encoded_group']]))

            graphs.append(data)
            labels.append(row['encoded_group'])

    # Convert labels to tensor
    labels = torch.tensor(labels, dtype=torch.long)
    return graphs, labels

def load_graphs_and_labels_with_epochs(gml_dir, feature_dir, metadata, n_epochs_per_sample):
    """
    Load epoched graphs and expand labels accordingly.
    Parameters:
        gml_dir: Directory containing the graph files
        feature_dir: Directory containing the feature files
        metadata: DataFrame containing subject metadata
        n_epochs_per_sample: List containing number of epochs per sample
    Returns:
        graphs: List of PyTorch Geometric Data objects
        labels: Tensor of expanded labels
    """
    graphs = []
    valid_labels = []  # Keep track of valid labels

    # Encode labels first
    label_encoder = LabelEncoder()
    metadata['encoded_group'] = label_encoder.fit_transform(metadata['Group'])

    # Get original labels
    original_labels = metadata['encoded_group'].values
    
    # Validate directories exist
    if not os.path.exists(gml_dir):
        raise ValueError(f"Graph directory does not exist: {gml_dir}")
    if not os.path.exists(feature_dir):
        raise ValueError(f"Feature directory does not exist: {feature_dir}")

    print(f"Processing {len(metadata)} subjects with {sum(n_epochs_per_sample)} total epochs")
    
    for idx, (_, row) in enumerate(metadata.iterrows()):
        if idx >= len(n_epochs_per_sample):
            print(f"Warning: n_epochs_per_sample list is shorter than metadata. Stopping at index {idx}")
            break
            
        n_epochs = n_epochs_per_sample[idx]
        subject_id = row['subject.id']
        
        print(f"Processing subject {subject_id} with {n_epochs} epochs")
        
        for epoch_idx in range(n_epochs):
            gml_file = os.path.join(gml_dir, f"{subject_id}_epoch{epoch_idx}_graph.gml")
            feature_file = os.path.join(feature_dir, f"{subject_id}_epoch{epoch_idx}_features.json")

            if not os.path.exists(gml_file):
                print(f"Warning: Missing graph file: {gml_file}")
                continue
            if not os.path.exists(feature_file):
                print(f"Warning: Missing feature file: {feature_file}")
                continue

            try:
                # Load graph
                graph = nx.read_gml(gml_file)

                # Load features
                with open(feature_file, 'r') as f:
                    features = json.load(f)

                # Create node mapping and convert to PyTorch Geometric format
                node_mapping = {node: idx for idx, node in enumerate(graph.nodes)}
                edge_index = torch.tensor([[node_mapping[u], node_mapping[v]] 
                                         for u, v in graph.edges], dtype=torch.long).t().contiguous()
                node_features = torch.tensor([features[node] for node in graph.nodes], 
                                           dtype=torch.float32)
                edge_attr = torch.tensor([data['weight'] for _, _, data in graph.edges(data=True)], 
                                       dtype=torch.float32)

                data = Data(
                    x=node_features,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    num_nodes=len(graph.nodes),
                    y=torch.tensor([original_labels[idx]])
                )

                graphs.append(data)
                valid_labels.append(original_labels[idx])
                
            except Exception as e:
                print(f"Error processing {gml_file}: {str(e)}")
                continue

    if not graphs:
        raise ValueError("No valid graphs were loaded")

    # Convert valid labels to tensor
    labels = torch.tensor(valid_labels, dtype=torch.long)
    
    print(f"Successfully loaded {len(graphs)} graphs")
    return graphs, labels 

class GATModel(Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
       
        super(GATModel, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)  # First attention layer
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)  # Second layer

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of the GAT model.
        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.
            edge_attr (torch.Tensor): Edge attributes (weights).
        Returns:
            torch.Tensor: Log-softmax probabilities for each class.
        """
        x = self.gat1(x, edge_index, edge_attr)  # First GAT layer
        x = F.elu(x)  # Apply activation
        x = self.gat2(x, edge_index, edge_attr)  # Second GAT layer
        return F.log_softmax(x, dim=1)  # Output class probabilities

 

class GradCamGAT:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward hook on the target layer
        self.target_layer.register_forward_hook(self.save_activation)

    def save_gradient(self, grad):
        # Save gradients during backward pass
        self.gradients = grad

    def save_activation(self, module, input, output):
        # Save activations during forward pass
        self.activations = output

    def forward(self, x, edge_index, edge_attr, batch):
        # Forward pass through the target layer
        node_output = self.target_layer(x, edge_index, edge_attr)
        node_output.register_hook(self.save_gradient)  # Register gradient hook
        graph_output = global_mean_pool(node_output, batch)  # Global pooling
        return graph_output, node_output

    def backward(self, output, class_idx=None):
        # Backward pass to compute gradients for a specific class
        if class_idx is None:
            class_idx = output.argmax(dim=1)  # Use the predicted class if no class is specified
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1  # Create one-hot vector for the target class
        output.backward(gradient=one_hot, retain_graph=True)

    def get_node_importance(self):
        # Compute node importance using gradients and activations
        if self.gradients is None or self.activations is None:
            raise ValueError("Gradients or activations have not been computed. Did you call forward and backward?")
        gradients = self.gradients.mean(dim=1, keepdim=True)  # Average gradients over nodes
        activations = self.activations  # Use saved activations
        node_importance = (gradients * activations).sum(dim=1)  # Grad-CAM formula
        return node_importance.detach()

"""# Nigerian dataset
 
### GAT model
"""

 
"""Processing all EEG files, create graphs and feature files."""

#process_all_files(ni_df, "output4/graphs_nigeria", "output4/features_nigeria")

n_epochs_per_sample = process_all_files_with_epochs(
    ni_df,
    "output4/graphs_nigeria_epoched",
    "output4/features_nigeria_epoched",
    duration=5,
    overlap=0
)
 

  
graphs, labels = load_graphs_and_labels_with_epochs(
    "output4/graphs_nigeria_epoched",
    "output4/features_nigeria_epoched",
    ni_df,
    n_epochs_per_sample
)
 
# Stratified split to ensure both classes are represented in train and test sets
train_graphs, test_graphs, train_labels, test_labels = train_test_split(
    graphs, labels, test_size=0.2, stratify=labels, random_state=42
)


print("Train class distribution:", Counter(train_labels.numpy()))
print("Test class distribution:", Counter(test_labels.numpy()))

"""Setting up the DataLoader and Model Initialization"""

# Create DataLoader
train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)

# Initialize model
model = GATModel(in_channels=5, hidden_channels=8, out_channels=2)  # Binary classification
optimizer = Adam(model.parameters(), lr=0.01)
 

for epoch in range(500):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()

        # Forward pass
        node_output = model(data.x, data.edge_index, data.edge_attr)

        # Aggregate node outputs to graph-level predictions using batch information
        graph_output = global_mean_pool(node_output, data.batch)

        # Compute loss
        loss = F.nll_loss(graph_output, data.y)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss}")
 

# Evaluate model on test set (with graph-level pooling)
model.eval()
all_preds = []
all_labels = []
all_probs = []


for data in test_graphs:
    # Forward pass
    node_output = model(data.x, data.edge_index, data.edge_attr)

    # Aggregate node outputs to graph-level predictions using batch information
    graph_prediction = global_mean_pool(node_output, data.batch)

    # Get predicted class
    preds = graph_prediction.argmax(dim=1)

    # Store predicted probabilities for the positive class (class 1)
    probs = torch.softmax(graph_prediction, dim=1)[:, 1].detach().numpy()  # Probabilities for class 1

    all_preds.append(preds.item())
    all_labels.append(data.y.item())
    all_probs.extend(probs)  # Append probabilities

# Generate classification report
print(classification_report(all_labels, all_preds, target_names=['Control', 'Epilepsy']))

# Convert lists to numpy arrays
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

# Alternatively, calculate ROC AUC score directly
roc_auc_score_value = roc_auc_score(all_labels, all_probs)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Print AUC value
print(f"AUC: {roc_auc:.2f}")
 

def get_attention_weights(model, data):
    with torch.no_grad():
        # Perform forward pass and retrieve attention weights
        _, attention_weights = model.gat1(data.x, data.edge_index, return_attention_weights=True)
    return attention_weights

# Iterate through multiple graphs and visualize attention weights
for idx, data in enumerate(test_graphs[:5]):  # Analyze first 5 graphs
    attention_weights = get_attention_weights(model, data)

    # Convert attention weights to NumPy
    attention_weights = attention_weights[1].numpy()  # Use [1] to get the actual attention weights

    # Flatten the attention weights for histogram
    flattened_weights = attention_weights.flatten()

    # Plot histogram
    plt.hist(flattened_weights, bins=20, color='blue', alpha=0.7)
    plt.xlabel("Attention Weights")
    plt.ylabel("Frequency")
    plt.title(f"Attention Weights for Graph {idx + 1}")
    plt.show()


 

# Example usage
data = test_graphs[0]  # Select one graph for analysis
batch = torch.zeros(data.x.size(0), dtype=torch.long)  # All nodes belong to the same graph

# Create Grad-CAM instance
grad_cam = GradCamGAT(model, target_layer=model.gat1)  # Use the first GAT layer as the target

# Forward pass through Grad-CAM
graph_output, node_output = grad_cam.forward(data.x, data.edge_index, data.edge_attr, batch)

# Backward pass to compute gradients
predicted_class = graph_output.argmax(dim=1).item()  # Get the predicted class
grad_cam.backward(graph_output, class_idx=predicted_class)

# Compute node importance
node_importance = grad_cam.get_node_importance()

# Visualize node importance
plt.bar(range(len(node_importance)), node_importance.numpy(), color='green')
plt.xlabel("Node Index")
plt.ylabel("Importance")
plt.title("Node Importance via Grad-CAM")
plt.show()
  
n_epochs_per_sample = process_all_files_with_epochs(
    ni_df,
    "output4/graphs_gb_epoched",
    "output4/features_gb_epoched",
    duration=5,
    overlap=0
)

"""Similarly, let's load one sample graph"""

import networkx as nx

 
"""### GAT model"""

 
# Load data
#graphs, labels = load_graphs_and_labels_with_features(gml_dir, feature_dir, gb_df)

# Load the epoched data with properly expanded labels
graphs, labels = load_graphs_and_labels_with_epochs(
    "output4/graphs_gb_epoched",
    "output4/features_gb_epoched",
    ni_df,
    n_epochs_per_sample
)
 
# Stratified split to ensure both classes are represented in train and test sets
train_graphs, test_graphs, train_labels, test_labels = train_test_split(
    graphs, labels, test_size=0.2, stratify=labels, random_state=42
)


print("Train class distribution:", Counter(train_labels.numpy()))
print("Test class distribution:", Counter(test_labels.numpy()))
 
# Create DataLoader
train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)

# Initialize model
model = GATModel(in_channels=5, hidden_channels=8, out_channels=2)  # Binary classification
optimizer = Adam(model.parameters(), lr=0.01)

for epoch in range(50):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()

        # Forward pass
        node_output = model(data.x, data.edge_index, data.edge_attr)

        # Aggregate node outputs to graph-level predictions using batch information
        graph_output = global_mean_pool(node_output, data.batch)

        # Compute loss
        loss = F.nll_loss(graph_output, data.y)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss}")

# Evaluate model on test set (with graph-level pooling)
model.eval()
all_preds = []
all_labels = []
all_probs = []

for data in test_graphs:
    # Forward pass
    node_output = model(data.x, data.edge_index, data.edge_attr)

    # Aggregate node outputs to graph-level predictions using batch information
    graph_prediction = global_mean_pool(node_output, data.batch)

    # Get predicted class
    preds = graph_prediction.argmax(dim=1)


    # Store predicted probabilities for the positive class (class 1)
    probs = torch.softmax(graph_prediction, dim=1)[:, 1].detach().numpy()  # Probabilities for class 1


    all_preds.append(preds.item())
    all_labels.append(data.y.item())

    all_probs.extend(probs)  # Append probabilities

# Generate classification report
print(classification_report(all_labels, all_preds, target_names=['Control', 'Epilepsy']))
 

# Convert lists to numpy arrays
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

# Alternatively, calculate ROC AUC score directly
roc_auc_score_value = roc_auc_score(all_labels, all_probs)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Print AUC value
print(f"AUC: {roc_auc:.2f}")

"""### Interpretability"""

# Extract attention weights
def get_attention_weights(model, data):
    with torch.no_grad():
        # Perform forward pass and retrieve attention weights
        _, attention_weights = model.gat1(data.x, data.edge_index, return_attention_weights=True)
    return attention_weights

# Iterate through multiple graphs and visualize attention weights
for idx, data in enumerate(test_graphs[:7]):  # Analyze first 5 graphs
    attention_weights = get_attention_weights(model, data)

    # Convert attention weights to NumPy
    attention_weights = attention_weights[1].numpy()  # Use [1] to get the actual attention weights

    # Flatten the attention weights for histogram
    flattened_weights = attention_weights.flatten()

    # Plot histogram
    plt.hist(flattened_weights, bins=20, color='blue', alpha=0.7)
    plt.xlabel("Attention Weights")
    plt.ylabel("Frequency")
    plt.title(f"Attention Weights for Graph {idx + 1}")
    plt.show()

 

# Example usage
data = test_graphs[0]  # Select one graph for analysis
batch = torch.zeros(data.x.size(0), dtype=torch.long)  # All nodes belong to the same graph

# Create Grad-CAM instance
grad_cam = GradCamGAT(model, target_layer=model.gat1)  # Use the first GAT layer as the target

# Forward pass through Grad-CAM
graph_output, node_output = grad_cam.forward(data.x, data.edge_index, data.edge_attr, batch)

# Backward pass to compute gradients
predicted_class = graph_output.argmax(dim=1).item()  # Get the predicted class
grad_cam.backward(graph_output, class_idx=predicted_class)

# Compute node importance
node_importance = grad_cam.get_node_importance()

# Visualize node importance
plt.bar(range(len(node_importance)), node_importance.numpy(), color='green')
plt.xlabel("Node Index")
plt.ylabel("Importance")
plt.title("Node Importance via Grad-CAM")
plt.show()
 
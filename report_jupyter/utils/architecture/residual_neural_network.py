# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Data processing and utilities
import numpy as np
from tqdm import tqdm
from IPython.display import clear_output

# Metrics and visualization
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import imageio

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_channels, in_channels),
            nn.Dropout(0.3)
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x
        out = self.block(x)
        out += identity  # Skip connection
        out = self.relu(out)
        return out

class ResidualNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, learning_rate: float = 1e-3):
        """
        Initialize residual neural network model.
        
        Args:
            input_dim (int): Number of input features
            learning_rate (float): Learning rate for optimization
        """
        super().__init__()
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() 
                                 else "mps" if torch.backends.mps.is_available() 
                                 else "cpu")
        print(f"Using device: {self.device}")
        
        # Initial layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Residual blocks
        self.res_block1 = ResidualBlock(128)
        self.res_block2 = ResidualBlock(128)
        self.res_block3 = ResidualBlock(128)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 8),
            nn.Softmax(dim=1)
        )
        
        # Move model to device
        self.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', 
                                                            factor=0.5, patience=1)
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.output_layer(x)
        return x
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
           X_val: np.ndarray, y_val: np.ndarray,
           batch_size: int = 32, epochs: int = 100) -> tuple[list[float], list[float]]:
        """
        Train the neural network model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation labels
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
            
        Returns:
            tuple[list[float], list[float]]: Training and validation losses per epoch
        """
        # Convert string labels to numeric indices
        unique_labels = np.unique(np.concatenate([y_train, y_val]))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        y_train_numeric = np.array([label_to_idx[label] for label in y_train])
        y_val_numeric = np.array([label_to_idx[label] for label in y_val])
        
        # Store mapping for later use
        self.label_to_idx = label_to_idx
        self.idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        
        # Calculate class weights
        class_counts = np.bincount(y_train_numeric)
        total_samples = len(y_train_numeric)
        class_weights = torch.FloatTensor(total_samples / (len(class_counts) * class_counts)).to(self.device)
        
        # Initialize weighted loss criterion
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Convert numpy arrays to tensors
        try:
            X = torch.FloatTensor(X_train.astype(np.float32)).to(self.device)
            y = torch.LongTensor(y_train_numeric).to(self.device)
            X_val_tensor = torch.FloatTensor(X_val.astype(np.float32)).to(self.device)
            y_val_tensor = torch.LongTensor(y_val_numeric).to(self.device)
        except ValueError as e:
            raise ValueError("Input features must be numeric. Check that X_train and X_val contain only numbers.") from e
        
        # Create dataset and dataloader
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        # For saving animation frames
        frames = []
        
        # Training loop with progress bar
        progress_bar = tqdm(range(epochs), desc="Training")
        for epoch in progress_bar:
            # Training phase
            self.train()
            epoch_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for batch_X, batch_y in dataloader:
                # Forward pass
                outputs = self(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Calculate training accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_train += batch_y.size(0)
                correct_train += (predicted == batch_y).sum().item()
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                epoch_loss += loss.item()
            
            # Average training loss and accuracy for the epoch
            avg_train_loss = epoch_loss / len(dataloader)
            train_accuracy = 100 * correct_train / total_train
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)
            
            # Validation phase
            self.eval()
            with torch.no_grad():
                val_outputs = self(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor)
                
                # Calculate validation accuracy
                _, predicted = torch.max(val_outputs.data, 1)
                val_accuracy = 100 * (predicted == y_val_tensor).sum().item() / len(y_val_tensor)
                
                val_losses.append(val_loss.item())
                val_accuracies.append(val_accuracy)
            
            # Update learning rate based on validation loss
            self.scheduler.step(val_loss)
            
            # Clear output and plot updated metrics
            clear_output(wait=True)
            
            # Create figure with two subplots
            fig = plt.figure(figsize=(12, 5))
            
            # Plot losses
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            
            # Plot accuracies
            plt.subplot(1, 2, 2)
            plt.plot(train_accuracies, label='Training Accuracy')
            plt.plot(val_accuracies, label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.title('Training and Validation Accuracy')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('./visualisations/residual_neural_network_training_plot.png')
            plt.show()
            
            # Save current figure to frames list
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)
            
            plt.close()
            
            # Update progress bar description with epoch metrics
            progress_bar.set_postfix({
                'train_loss': f'{avg_train_loss:.4f}',
                'val_loss': f'{val_loss.item():.4f}',
                'train_acc': f'{train_accuracy:.2f}%',
                'val_acc': f'{val_accuracy:.2f}%'
            })
        
        # Save animation as GIF
        imageio.mimsave('./visualisations/residual_neural_network_training.gif', frames, fps=2)
                
        return train_losses, val_losses
    
    def predict(self, X: np.ndarray, y_test: np.ndarray = None, display_metrics: bool = True) -> np.ndarray:
        """
        Make predictions using the trained model and optionally display performance metrics.
        
        Args:
            X (np.ndarray): Input features
            y_test (np.ndarray, optional): True labels for computing metrics
            display_metrics (bool): Whether to display classification metrics
            
        Returns:
            np.ndarray: Class predictions as original category labels
        """
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            probabilities = self(X_tensor)
            predictions = torch.argmax(probabilities, dim=1).cpu().numpy()
            # Convert numeric predictions back to original labels
            predictions = np.array([self.idx_to_label[idx] for idx in predictions])
            
        if display_metrics and y_test is not None:
            print("\nClassification Report:")
            print(classification_report(y_test, predictions, zero_division=0))
            
            # Create confusion matrix
            plt.figure(figsize=(10,8))
            cm = confusion_matrix(y_test, predictions)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=sorted(set(y_test)),
                       yticklabels=sorted(set(y_test)))
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig('./visualisations/residual_neural_network_confusion_matrix.png')
            plt.show()
            
        return None
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions for all classes.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Probability predictions for each class (shape: n_samples x n_classes)
        """
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            probabilities = self(X_tensor)
        return probabilities.cpu().numpy()

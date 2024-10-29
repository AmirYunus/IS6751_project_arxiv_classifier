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

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, learning_rate: float = 1e-3):
        """
        Initialize convolutional neural network model.
        
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
        
        # Reshape input features into a square-like image
        self.side_length = int(np.ceil(np.sqrt(input_dim)))
        self.pad_size = self.side_length * self.side_length - input_dim
        
        # Convolutional layers with reduced complexity
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3)
        )
        
        # Calculate size of flattened features after convolutions
        self.flat_features = self._get_flat_features()
        
        # Fully connected layers with reduced complexity
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 8),
            nn.Softmax(dim=1)
        )
        
        # Move model to device
        self.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', 
                                                            factor=0.5, patience=1)
        
    def _get_flat_features(self):
        # Helper function to calculate flattened size
        x = torch.randn(1, 1, self.side_length, self.side_length)
        x = self.conv_layers(x)
        return int(np.prod(x.size()[1:]))
        
    def forward(self, x):
        # Pad input if necessary
        if self.pad_size > 0:
            x = torch.nn.functional.pad(x, (0, self.pad_size))
        
        # Reshape to square image format
        batch_size = x.size(0)
        x = x.view(batch_size, 1, self.side_length, self.side_length)
        
        # Forward pass through conv layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(batch_size, -1)
        
        # Forward pass through FC layers
        x = self.fc_layers(x)
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
        
        # Convert numpy arrays to tensors in batches
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train.astype(np.float32)),
            torch.LongTensor(y_train_numeric)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val.astype(np.float32)),
            torch.LongTensor(y_val_numeric)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
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
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
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
            avg_train_loss = epoch_loss / len(train_loader)
            train_accuracy = 100 * correct_train / total_train
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self(batch_X)
                    val_loss += self.criterion(outputs, batch_y).item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += batch_y.size(0)
                    correct_val += (predicted == batch_y).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct_val / total_val
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            
            # Update learning rate based on validation loss
            self.scheduler.step(avg_val_loss)
            
            # Clear output before showing new plot
            clear_output(wait=True)
            
            # Create figure for current epoch
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
            plt.savefig('./visualisations/convolutional_neural_network_training_plot.png')
            
            # Display current plot
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
                'val_loss': f'{avg_val_loss:.4f}',
                'train_acc': f'{train_accuracy:.2f}%',
                'val_acc': f'{val_accuracy:.2f}%'
            })
        
        # Save animation as GIF
        imageio.mimsave('./visualisations/convolutional_neural_network_training.gif', frames, fps=2)
                
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
        predictions = []
        
        # Process in batches
        dataset = TensorDataset(torch.FloatTensor(X))
        dataloader = DataLoader(dataset, batch_size=32)
        
        with torch.no_grad():
            for batch_X, in dataloader:
                batch_X = batch_X.to(self.device)
                batch_probs = self(batch_X)
                batch_preds = torch.argmax(batch_probs, dim=1).cpu().numpy()
                predictions.extend([self.idx_to_label[idx] for idx in batch_preds])
        
        predictions = np.array(predictions)
            
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
            plt.savefig('./visualisations/convolutional_neural_network_confusion_matrix.png')
            plt.show()
            plt.close()
            
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions for all classes.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Probability predictions for each class (shape: n_samples x n_classes)
        """
        self.eval()
        probabilities = []
        
        # Process in batches
        dataset = TensorDataset(torch.FloatTensor(X))
        dataloader = DataLoader(dataset, batch_size=32)
        
        with torch.no_grad():
            for batch_X, in dataloader:
                batch_X = batch_X.to(self.device)
                batch_probs = self(batch_X)
                probabilities.append(batch_probs.cpu().numpy())
                
        return np.vstack(probabilities)

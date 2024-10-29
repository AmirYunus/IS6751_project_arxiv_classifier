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

class AutoencoderNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8, learning_rate: float = 1e-3):
        """
        Initialize autoencoder model.
        
        Args:
            input_dim (int): Number of input features
            latent_dim (int): Size of latent space dimension
            learning_rate (float): Learning rate for optimization
        """
        super().__init__()
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() 
                                 else "mps" if torch.backends.mps.is_available() 
                                 else "cpu")
        print(f"Using device: {self.device}")
        
        # Store input dimension
        self.input_dim = input_dim
        
        # Reshape input features into a square-like image
        self.side_length = int(np.ceil(np.sqrt(input_dim)))
        self.pad_size = self.side_length * self.side_length - input_dim
        
        # Encoder layers with smaller feature maps
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3)
        )
        
        # Calculate size of flattened features after convolutions
        self.flat_features = self._get_flat_features()
        
        # Latent space with reduced dimensions
        self.fc_encoder = nn.Sequential(
            nn.Linear(self.flat_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, latent_dim)
        )
        
        # Decoder layers with reduced dimensions
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.flat_features)
        )
        
        # Decoder conv layers with reduced feature maps
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.3),
            
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.3),
            
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 8),
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
        x = self.encoder(x)
        return int(np.prod(x.size()[1:]))
        
    def encode(self, x):
        # Process data in smaller batches
        batch_size = 32
        dataset = TensorDataset(x)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        encoded_batches = []
        for batch_x, in dataloader:
            # Pad input if necessary
            if self.pad_size > 0:
                batch_x = torch.nn.functional.pad(batch_x, (0, self.pad_size))
            
            # Reshape to square image format
            batch_x = batch_x.view(-1, 1, self.side_length, self.side_length)
            
            # Forward pass through encoder
            batch_x = self.encoder(batch_x)
            batch_x = batch_x.view(batch_x.size(0), -1)
            encoded_batches.append(self.fc_encoder(batch_x))
            
        return torch.cat(encoded_batches, dim=0)
        
    def decode(self, z):
        # Process data in smaller batches
        batch_size = 32
        dataset = TensorDataset(z)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        decoded_batches = []
        for batch_z, in dataloader:
            # Decode from latent space
            batch_x = self.fc_decoder(batch_z)
            batch_x = batch_x.view(-1, 64, self.side_length//(2**3), self.side_length//(2**3))
            batch_x = self.decoder_conv(batch_x)
            batch_x = batch_x.view(batch_x.size(0), -1)
            
            # Remove padding if necessary
            if self.pad_size > 0:
                batch_x = batch_x[:, :-self.pad_size]
            decoded_batches.append(batch_x)
            
        return torch.cat(decoded_batches, dim=0)
        
    def forward(self, x):
        z = self.encode(x)
        reconstruction = self.decode(z)
        classification = self.classifier(z)
        return reconstruction, classification
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            batch_size: int = 32, epochs: int = 100) -> tuple[list[float], list[float]]:
        """
        Train the autoencoder model.
        
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
        
        # Initialize loss criteria
        self.recon_criterion = nn.MSELoss()
        self.class_criterion = nn.CrossEntropyLoss(weight=class_weights)
        
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
                recon, outputs = self(batch_X)
                
                # Calculate losses
                # Ensure reconstruction has same size as input by truncating or padding
                if recon.size(1) > batch_X.size(1):
                    recon = recon[:, :batch_X.size(1)]
                elif recon.size(1) < batch_X.size(1):
                    recon = torch.nn.functional.pad(recon, (0, batch_X.size(1) - recon.size(1)))
                
                recon_loss = self.recon_criterion(recon, batch_X)
                class_loss = self.class_criterion(outputs, batch_y)
                loss = recon_loss + class_loss
                
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
                # Process validation data in batches
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
                
                val_loss = 0.0
                correct_val = 0
                total_val = 0
                
                for batch_X_val, batch_y_val in val_dataloader:
                    recon_val, val_outputs = self(batch_X_val)
                    
                    # Ensure reconstruction has same size as input
                    if recon_val.size(1) > batch_X_val.size(1):
                        recon_val = recon_val[:, :batch_X_val.size(1)]
                    elif recon_val.size(1) < batch_X_val.size(1):
                        recon_val = torch.nn.functional.pad(recon_val, (0, batch_X_val.size(1) - recon_val.size(1)))
                    
                    recon_loss = self.recon_criterion(recon_val, batch_X_val)
                    class_loss = self.class_criterion(val_outputs, batch_y_val)
                    batch_val_loss = recon_loss + class_loss
                    val_loss += batch_val_loss.item()
                    
                    # Calculate validation accuracy
                    _, predicted = torch.max(val_outputs.data, 1)
                    total_val += batch_y_val.size(0)
                    correct_val += (predicted == batch_y_val).sum().item()
                
                avg_val_loss = val_loss / len(val_dataloader)
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
            plt.savefig('./visualisations/autoencoder_neural_network_training_plot.png')
            
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
        imageio.mimsave('./visualisations/autoencoder_neural_network_training.gif', frames, fps=2)
                
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
            # Process test data in batches
            batch_size = 32
            X_tensor = torch.FloatTensor(X).to(self.device)
            test_dataset = TensorDataset(X_tensor)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
            
            all_predictions = []
            for batch_X, in test_dataloader:
                _, probabilities = self(batch_X)
                predictions = torch.argmax(probabilities, dim=1).cpu().numpy()
                all_predictions.append(predictions)
            
            predictions = np.concatenate(all_predictions)
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
            plt.savefig('./visualisations/autoencoder_neural_network_confusion_matrix.png')
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
        with torch.no_grad():
            # Process test data in batches
            batch_size = 32
            X_tensor = torch.FloatTensor(X).to(self.device)
            test_dataset = TensorDataset(X_tensor)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
            
            all_probabilities = []
            for batch_X, in test_dataloader:
                _, probabilities = self(batch_X)
                all_probabilities.append(probabilities.cpu().numpy())
            
        return np.concatenate(all_probabilities, axis=0)

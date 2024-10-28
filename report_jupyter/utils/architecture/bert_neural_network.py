# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertTokenizer

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

class BertNeuralNetwork(nn.Module):
    def __init__(self, num_classes: int = 8, learning_rate: float = 1e-7):
        """
        Initialize BERT model for classification.
        
        Args:
            num_classes (int): Number of output classes
            learning_rate (float): Learning rate for optimization
        """
        super().__init__()
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() 
                                 else "mps" if torch.backends.mps.is_available() 
                                 else "cpu")
        print(f"Using device: {self.device}")
        
        # Load pre-trained BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)
        )
        
        # Move model to device
        self.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', 
                                                            factor=0.5, patience=1)
        
    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Pass through classifier
        return self.classifier(pooled_output)
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
           X_val: np.ndarray, y_val: np.ndarray,
           batch_size: int = 8, epochs: int = 100) -> tuple[list[float], list[float]]:
        """
        Train the BERT model.
        
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
        
        # Initialize loss criterion
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Convert numpy arrays to lists of strings
        X_train_list = [str(x) for x in X_train]
        X_val_list = [str(x) for x in X_val]
        
        # Tokenize input data
        train_encodings = self.tokenizer(X_train_list, truncation=True, padding=True, max_length=512)
        val_encodings = self.tokenizer(X_val_list, truncation=True, padding=True, max_length=512)
        
        # Convert to tensors
        train_input_ids = torch.tensor(train_encodings['input_ids'])
        train_attention_mask = torch.tensor(train_encodings['attention_mask'])
        train_labels = torch.tensor(y_train_numeric)
        
        val_input_ids = torch.tensor(val_encodings['input_ids']).to(self.device)
        val_attention_mask = torch.tensor(val_encodings['attention_mask']).to(self.device)
        val_labels = torch.tensor(y_val_numeric).to(self.device)
        
        # Create dataset and dataloader
        train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        frames = []
        
        # Training loop with progress bar
        progress_bar = tqdm(range(epochs), desc="Training")
        for epoch in progress_bar:
            # Training phase
            self.train()
            epoch_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for batch_input_ids, batch_attention_mask, batch_labels in train_dataloader:
                # Move batch to device
                batch_input_ids = batch_input_ids.to(self.device)
                batch_attention_mask = batch_attention_mask.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                outputs = self(batch_input_ids, batch_attention_mask)
                
                # Calculate loss
                loss = self.criterion(outputs, batch_labels)
                
                # Calculate training accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_train += batch_labels.size(0)
                correct_train += (predicted == batch_labels).sum().item()
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                epoch_loss += loss.item()
                
                # Clear GPU memory
                del outputs, loss, predicted
                batch_input_ids = batch_input_ids.cpu()
                batch_attention_mask = batch_attention_mask.cpu()
                batch_labels = batch_labels.cpu()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Average training loss and accuracy for the epoch
            avg_train_loss = epoch_loss / len(train_dataloader)
            train_accuracy = 100 * correct_train / total_train
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                val_outputs = self(val_input_ids, val_attention_mask)
                val_loss = self.criterion(val_outputs, val_labels).item()
                
                _, predicted = torch.max(val_outputs.data, 1)
                total_val = val_labels.size(0)
                correct_val = (predicted == val_labels).sum().item()
                val_accuracy = 100 * correct_val / total_val
                
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
            
            # Update learning rate based on validation loss
            self.scheduler.step(val_loss)
            
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
            plt.savefig('./visualisations/bert_neural_network_training_plot.png')
            
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
                'val_loss': f'{val_loss:.4f}',
                'train_acc': f'{train_accuracy:.2f}%',
                'val_acc': f'{val_accuracy:.2f}%'
            })
        
        # Save animation as GIF
        imageio.mimsave('./visualisations/bert_neural_network_training.gif', frames, fps=2)
                
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
        batch_size = 4  # Reduced batch size for prediction
        
        # Convert numpy array to list of strings
        X_list = [str(x) for x in X]
        
        # Tokenize input data
        encodings = self.tokenizer(X_list, truncation=True, padding=True, max_length=512)
        input_ids = torch.tensor(encodings['input_ids'])
        attention_mask = torch.tensor(encodings['attention_mask'])
        
        with torch.no_grad():
            # Process in batches
            for i in range(0, len(X), batch_size):
                batch_input_ids = input_ids[i:i+batch_size].to(self.device)
                batch_attention_mask = attention_mask[i:i+batch_size].to(self.device)
                
                batch_probs = self(batch_input_ids, batch_attention_mask)
                batch_preds = torch.argmax(batch_probs, dim=1).cpu().numpy()
                predictions.extend(batch_preds)
                
                # Clear memory
                del batch_input_ids, batch_attention_mask, batch_probs
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
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
            plt.savefig('./visualisations/bert_neural_network_confusion_matrix.png')
            plt.show()
            plt.close()
            
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
        probabilities = []
        batch_size = 4  # Reduced batch size for prediction
        
        # Convert numpy array to list of strings
        X_list = [str(x) for x in X]
        
        # Tokenize input data
        encodings = self.tokenizer(X_list, truncation=True, padding=True, max_length=512)
        input_ids = torch.tensor(encodings['input_ids'])
        attention_mask = torch.tensor(encodings['attention_mask'])
        
        with torch.no_grad():
            # Process in batches
            for i in range(0, len(X), batch_size):
                batch_input_ids = input_ids[i:i+batch_size].to(self.device)
                batch_attention_mask = attention_mask[i:i+batch_size].to(self.device)
                
                batch_probs = self(batch_input_ids, batch_attention_mask)
                probabilities.append(batch_probs.cpu().numpy())
                
                # Clear memory
                del batch_input_ids, batch_attention_mask, batch_probs
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
        return np.vstack(probabilities)

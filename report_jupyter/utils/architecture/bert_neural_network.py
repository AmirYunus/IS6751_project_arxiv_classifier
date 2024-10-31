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
    def __init__(self, num_classes: int = 8, learning_rate: float = 2e-5, batch_size: int = 16):
        """
        Initialize BERT model for fine-tuning classification.
        
        Args:
            num_classes (int): Number of output classes
            learning_rate (float): Learning rate for optimization
            batch_size (int): Batch size for processing data
        """
        super().__init__()
        
        # Set device
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs!")
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        self.batch_size = batch_size
        
        # Load pre-trained BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Classification head
        self.classifier = nn.Linear(768, num_classes)
        
        # Move model to device and wrap with DataParallel if multiple GPUs
        self.to(self.device)
        if torch.cuda.device_count() > 1:
            self.bert = nn.DataParallel(self.bert)
            self.classifier = nn.DataParallel(self.classifier)
        
        # Initialize optimizer with different learning rates
        # Higher learning rate for classifier, lower for BERT layers
        optimizer_grouped_parameters = [
            {'params': self.bert.parameters(), 'lr': learning_rate},
            {'params': self.classifier.parameters(), 'lr': learning_rate * 10}
        ]
        
        self.optimizer = optim.AdamW(optimizer_grouped_parameters)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                            factor=0.5, patience=1)
        
    def _process_batch(self, batch_data, training=True):
        """Helper method to process a single batch of data"""
        batch_input_ids, batch_attention_mask = batch_data
        batch_input_ids = batch_input_ids.to(self.device)
        batch_attention_mask = batch_attention_mask.to(self.device)
        
        outputs = self(batch_input_ids, batch_attention_mask)
        
        # Clear memory
        del batch_input_ids, batch_attention_mask
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return outputs
        
    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Pass through classifier
        logits = self.classifier(pooled_output)
        return torch.softmax(logits, dim=1)
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
           X_val: np.ndarray, y_val: np.ndarray,
           epochs: int = 10) -> tuple[list[float], list[float]]:
        """
        Fine-tune the BERT model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels  
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation labels
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
        
        # Process data in batches
        train_data = self._prepare_data(X_train, y_train_numeric)
        val_data = self._prepare_data(X_val, y_val_numeric)
        
        train_dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=self.batch_size)
        
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        frames = []
        
        # Training loop with progress bar
        progress_bar = tqdm(range(epochs), desc="Training", leave=True)
        for epoch in progress_bar:
            # Training phase
            train_metrics = self._train_epoch(train_dataloader)
            train_losses.append(train_metrics['loss'])
            train_accuracies.append(train_metrics['accuracy'])
            
            # Validation phase
            val_metrics = self._validate_epoch(val_dataloader)
            val_losses.append(val_metrics['loss'])
            val_accuracies.append(val_metrics['accuracy'])
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Visualization
            frames.append(self._create_training_plot(train_losses, val_losses, 
                                                   train_accuracies, val_accuracies))
            
            # Update progress bar
            progress_bar.set_postfix({
                'train_loss': f"{train_metrics['loss']:.4f}",
                'val_loss': f"{val_metrics['loss']:.4f}",
                'train_acc': f"{train_metrics['accuracy']:.2f}%",
                'val_acc': f"{val_metrics['accuracy']:.2f}%"
            })
        
        # Save animation
        imageio.mimsave('./visualisations/bert_neural_network_training.gif', frames, fps=2)
        
        return train_losses, val_losses
    
    def _prepare_data(self, X: np.ndarray, y: np.ndarray = None):
        """Helper method to prepare data for training/inference"""
        X_list = [str(x) for x in X]
        encodings = self.tokenizer(X_list, truncation=True, padding=True, max_length=128)
        input_ids = torch.tensor(encodings['input_ids'])
        attention_mask = torch.tensor(encodings['attention_mask'])
        
        if y is not None:
            labels = torch.tensor(y)
            return TensorDataset(input_ids, attention_mask, labels)
        return TensorDataset(input_ids, attention_mask)
    
    def _train_epoch(self, dataloader):
        """Helper method to train for one epoch"""
        self.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(dataloader, desc="Training batches", leave=True)
        for batch in progress_bar:
            input_ids, attention_mask, labels = [x.to(self.device) for x in batch]
            
            self.optimizer.zero_grad()
            outputs = self(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({'batch_loss': f"{loss.item():.4f}"})
            
            # Clear memory
            del outputs, loss, predicted, input_ids, attention_mask, labels
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': 100 * correct / total
        }
    
    def _validate_epoch(self, dataloader):
        """Helper method to validate for one epoch"""
        self.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Validation batches", leave=True)
            for batch in progress_bar:
                input_ids, attention_mask, labels = [x.to(self.device) for x in batch]
                
                outputs = self(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({'batch_loss': f"{loss.item():.4f}"})
                
                # Clear memory
                del outputs, loss, predicted, input_ids, attention_mask, labels
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': 100 * correct / total
        }
    
    def _create_training_plot(self, train_losses, val_losses, train_accuracies, val_accuracies):
        """Helper method to create and save training plots"""
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
        
        # Convert plot to image for animation
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close()
        return image
    
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
        dataset = self._prepare_data(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        
        predictions = []
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Prediction batches", leave=True)
            for batch in progress_bar:
                outputs = self._process_batch(batch, training=False)
                batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
                predictions.extend([self.idx_to_label[idx] for idx in batch_preds])
                
                del outputs, batch_preds
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        predictions = np.array(predictions)
        
        if display_metrics and y_test is not None:
            self._display_metrics(y_test, predictions)
            
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions for all classes.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Probability predictions for each class
        """
        self.eval()
        dataset = self._prepare_data(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        
        probabilities = []
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Probability prediction batches", leave=True)
            for batch in progress_bar:
                outputs = self._process_batch(batch, training=False)
                probabilities.append(outputs.cpu().numpy())
                
                del outputs
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
        return np.vstack(probabilities)
    
    def _display_metrics(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Helper method to display classification metrics"""
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
        
        plt.figure(figsize=(10,8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=sorted(set(y_true)),
                   yticklabels=sorted(set(y_true)))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('./visualisations/bert_neural_network_confusion_matrix.png')
        plt.show()
        plt.close()

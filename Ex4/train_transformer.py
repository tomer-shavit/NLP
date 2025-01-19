import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm
import evaluate

# Import from exercise_blanks for data and constants
from data_loader import get_negated_polarity_examples, get_rare_words_examples
from exercise_blanks import (
    DataManager, TRAIN, VAL, TEST, W2V_SEQUENCE, evaluate_subset_accuracy,
    plot_train_val, get_available_device
)


class SentimentDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length=128):
        # Convert sentences to text format and get labels
        texts = [" ".join(sent.text) for sent in sentences]
        labels = [sent.sentiment_class for sent in sentences]
        
        # Tokenize texts
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
        self.labels = torch.tensor(labels, dtype=torch.long)  # Changed to long for classification

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item


def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(data_loader)


def evaluate_model(model, data_loader, device, metric):
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            
            predictions = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    
    # Calculate metrics
    metrics = metric.compute(predictions=all_predictions, references=all_labels)
    avg_loss = total_loss / len(data_loader)
    
    return avg_loss, metrics['accuracy']


def evaluate_subset(model, dataset, indices, device, batch_size=32):
    """Evaluate model on a subset of the data specified by indices."""
    # Create a subset of the dataset
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size)
    
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch['labels']).sum().item()
            total += len(predictions)
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def train_transformer(portion=1.0):
    """Main function to train the transformer model."""
    # Initialize DataManager
    dm = DataManager(data_type=W2V_SEQUENCE, batch_size=32)
    
    # Setup device
    device = get_available_device()
    
    # Model parameters
    model_name = "distilroberta-base"
    epochs = 3
    batch_size = 16
    learning_rate = 5e-5
    
    # Initialize model, tokenizer, and metric
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    metric = evaluate.load("accuracy")
    
    # Create datasets
    train_dataset = SentimentDataset(dm.sentences[TRAIN], tokenizer)
    val_dataset = SentimentDataset(dm.sentences[VAL], tokenizer)
    test_dataset = SentimentDataset(dm.sentences[TEST], tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Training phase
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        # Validation phase
        val_loss, val_accuracy = evaluate_model(model, val_loader, device, metric)
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)
        
        # Calculate training accuracy
        train_loss, train_accuracy = evaluate_model(model, train_loader, device, metric)
        train_accs.append(train_accuracy)
        
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Train Acc = {train_accuracy:.4f}")
        print(f"           Val Loss = {val_loss:.4f}, Val Acc = {val_accuracy:.4f}")
    
    # Final test evaluation
    test_loss, test_accuracy = evaluate_model(model, test_loader, device, metric)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Evaluate on special subsets
    negated_indices = get_negated_polarity_examples(dm.sentences[TEST])
    rare_indices = get_rare_words_examples(dm.sentences[TEST], dm.sentiment_dataset)
    
    neg_loss, neg_acc = evaluate_subset(model, test_dataset, negated_indices, device)
    rare_loss, rare_acc = evaluate_subset(model, test_dataset, rare_indices, device)
    print(f"Negated Polarity Accuracy: {neg_acc:.2%}")
    print(f"Rare Words Accuracy: {rare_acc:.2%}")
    
    # Plot metrics
    plot_train_val(train_losses, val_losses, train_accs, val_accs)


if __name__ == "__main__":
    train_transformer()
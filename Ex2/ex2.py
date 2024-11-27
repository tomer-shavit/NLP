

###################################################
# Exercise 2 - Natural Language Processing 67658  #
###################################################

import numpy as np

# subset of categories that we will use
category_dict = {'comp.graphics': 'computer graphics',
                 'rec.sport.baseball': 'baseball',
                 'sci.electronics': 'science, electronics',
                 'talk.politics.guns': 'politics, guns'
                 }

def get_data(categories=None, portion=1.):
    """
    Get data for given categories and portion
    :param portion: portion of the data to use
    :return:
    """
    # get data
    from sklearn.datasets import fetch_20newsgroups
    data_train = fetch_20newsgroups(categories=categories, subset='train', remove=('headers', 'footers', 'quotes'),
                                    random_state=21)
    data_test = fetch_20newsgroups(categories=categories, subset='test', remove=('headers', 'footers', 'quotes'),
                                   random_state=21)

    # train
    train_len = int(portion*len(data_train.data))
    x_train = np.array(data_train.data[:train_len])
    y_train = data_train.target[:train_len]
    # remove empty entries
    non_empty = x_train != ""
    x_train, y_train = x_train[non_empty].tolist(), y_train[non_empty].tolist()

    # test
    x_test = np.array(data_test.data)
    y_test = data_test.target
    non_empty = np.array(x_test) != ""
    x_test, y_test = x_test[non_empty].tolist(), y_test[non_empty].tolist()
    return x_train, y_train, x_test, y_test

def MLP_classification(portion=1., model=None):
    """
    Perform linear classification
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    ########### add your code here ###########
    vectorizer = TfidfVectorizer(max_features=2000)
    x_train = vectorizer.fit_transform(x_train).toarray()
    x_test = vectorizer.transform(x_test).toarray()

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    x_train, y_train = torch.tensor(x_train, dtype=torch.float32).to(device), torch.tensor(y_train,
                                                                                           dtype=torch.long).to(device)
    x_val, y_val = torch.tensor(x_val, dtype=torch.float32).to(device), torch.tensor(y_val, dtype=torch.long).to(device)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=32)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_accuracies = [], []
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                outputs = model(x_batch)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
        val_accuracies.append(correct / total)
        print(f"Epoch {epoch + 1}: Loss = {train_losses[-1]:.4f}, Val Accuracy = {val_accuracies[-1]:.4f}")

    return train_losses, val_accuracies
# Q1,2
def build_linear_model(input_dim, output_dim):
    """
    Build a linear model for classification.
    :param input_dim: Input dimension (e.g., number of features).
    :param output_dim: Output dimension (e.g., number of classes).
    :return: Linear model.
    """
    import torch.nn as nn
    return nn.Linear(input_dim, output_dim)


def build_mlp_model(input_dim, hidden_dim, output_dim):
    """
    Build an MLP model with one hidden layer.
    :param input_dim: Input dimension (e.g., number of features).
    :param hidden_dim: Dimension of the hidden layer.
    :param output_dim: Output dimension (e.g., number of classes).
    :return: MLP model.
    """
    import torch.nn as nn

    class MLPModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(MLPModel, self).__init__()
            self.hidden = nn.Linear(input_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, output_dim)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.hidden(x)
            x = self.relu(x)
            x = self.output(x)
            return x

    return MLPModel(input_dim, hidden_dim, output_dim)

def plot_metrics(portion, train_losses, val_accuracies, model_type):
    """
    Plot training loss and validation accuracy for each epoch.
    :param portion: Portion of the training data used.
    :param train_losses: List of training loss per epoch.
    :param val_accuracies: List of validation accuracy per epoch.
    :param model_type: Type of the model (e.g., "Linear" or "MLP").
    """
    import matplotlib.pyplot as plt
    import os

    # Ensure output directory exists
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_type} - Train Loss (Portion: {portion})')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{model_type} - Validation Accuracy (Portion: {portion})')
    plt.legend()

    plt.tight_layout()

    file_name = os.path.join(output_dir, f"{model_type}_metrics_portion_{portion}.png")
    plt.savefig(file_name)
    plt.close()
    print(f"Plot saved as {file_name}")

# Q3
def transformer_classification(portion=1.):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from torch.utils.data import DataLoader
    import evaluate
    from tqdm import tqdm

    class Dataset(torch.utils.data.Dataset):
        """
        Dataset for loading data
        """
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

        def __len__(self):
            return len(self.labels)

    def train_epoch(model, data_loader, optimizer, dev='cpu'):
        """
        Perform an epoch of training of the model with the optimizer
        :param model:
        :param data_loader:
        :param optimizer:
        :param dev:
        :return: Average loss over the epoch
        """
        model.train()
        total_loss = 0.
        # iterate over batches
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(dev)
            attention_mask = batch['attention_mask'].to(dev)
            labels = batch['labels'].to(dev)
            ########### add your code here ###########

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(data_loader)

    def evaluate_model(model, data_loader, dev='cpu', metric=None):
        model.eval()
        # metric.reset()
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(dev)
            attention_mask = batch['attention_mask'].to(dev)
            labels = batch['labels'].to(dev)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            metric.add_batch(predictions=predictions, references=labels)

        return metric.compute()['accuracy']

    def plot_transformer_metrics(portion, train_losses, val_accuracies):
        """
        Plot training loss and validation accuracy for each epoch and save the plot.
        """
        import matplotlib.pyplot as plt
        import os

        output_dir = "plots"
        os.makedirs(output_dir, exist_ok=True)

        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(12, 6))

        # Plot Training Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Transformer - Train Loss (Portion: {portion})')
        plt.legend()

        # Plot Validation Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='green')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Transformer - Validation Accuracy (Portion: {portion})')
        plt.legend()

        plt.tight_layout()
        file_name = os.path.join(output_dir, f"Transformer_metrics_portion_{portion}.png")
        plt.savefig(file_name)
        plt.close()
        print(f"Plot saved as {file_name}")

    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    # Parameters
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_labels = len(category_dict)
    epochs = 3
    batch_size = 16
    learning_rate = 5e-5

    # Model, tokenizer, and metric
    model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base', num_labels=num_labels).to(dev)
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    metric = evaluate.load("accuracy")

    # Datasets and DataLoaders
    train_dataset = Dataset(tokenizer(x_train, truncation=True, padding=True), y_train)
    val_dataset = Dataset(tokenizer(x_test, truncation=True, padding=True), y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    ########### add your code here ###########
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_losses, val_accuracies = [], []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, dev)
        val_accuracy = evaluate_model(model, val_loader, dev, metric)

        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Validation Accuracy = {val_accuracy:.4f}")

    plot_transformer_metrics(portion, train_losses, val_accuracies)


if __name__ == "__main__":
    portions = [0.1, 0.2, 0.5, 1.]
    # Q1 - single layer MLP
    # print("\nRunning experiments with Linear Classifier")
    # for portion in portions:
    #     model = build_linear_model(input_dim=2000, output_dim=len(category_dict))
    #     train_losses, val_accuracies = MLP_classification(portion=portion, model=model)
    #     plot_metrics(portion, train_losses, val_accuracies, model_type="Linear")

    # Q2 - multi-layer MLP
    # print("\nRunning experiments with MLP Classifier")
    # for portion in portions:
    #     model = build_mlp_model(input_dim=2000, hidden_dim=500, output_dim=len(category_dict))
    #     train_losses, val_accuracies = MLP_classification(portion=portion, model=model)
    #     plot_metrics(portion, train_losses, val_accuracies, model_type="MLP")

    # Q3 - Transformer
    print("\nTransformer results:")
    for p in portions[:2]:
        print(f"Portion: {p}")
        transformer_classification(portion=p)

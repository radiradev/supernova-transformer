import torch
import torch.nn as nn
import schedulefree
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model import TransformerNet
from dataset import Dataset

# Hyperparameters
BATCH_SIZE = 2048
NUM_EPOCHS = 50
LEARNING_RATE = 5e-4
NUM_ENCODER_LAYERS = 3
EMB_SIZE = 128
NUM_HEAD = 2
NUM_CLASSES = 1  # Binary classification

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_model():
    # Load and split the data
    dataset = Dataset(es_path="data/elastic_scatter.npy", cc_path="data/charged_current.npy")
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    # Initialize the model and move it to GPU
    model = TransformerNet(num_encoder_layers=NUM_ENCODER_LAYERS, 
                           emb_size=EMB_SIZE, 
                           num_head=NUM_HEAD, 
                           num_classes=NUM_CLASSES).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_data, batch_labels in train_loader:
            # Move data to GPU
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs.squeeze(), batch_labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_true = []
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                # Move data to GPU
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                
                outputs = model(batch_data)
                loss = criterion(outputs.squeeze(), batch_labels.float())
                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(outputs).squeeze().cpu().numpy() > 0.5)
                val_true.extend(batch_labels.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(val_true, val_preds)
        precision = precision_score(val_true, val_preds)
        recall = recall_score(val_true, val_preds)
        f1 = f1_score(val_true, val_preds)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("--------------------------")

    # Save the model
    torch.save(model.state_dict(), "transformer_model.pth")
    print("Training completed. Model saved.")

if __name__ == "__main__":
    train_model()
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import DualGridDataset3D
from model import DualTowerGridCNN3D

def train_3d_cnn(X_train_south, X_train_north, y_train_all, X_pred_south, X_pred_north, epochs=20, batch_size=16, lr=0.005):
    
    # [CRITICAL FIX] Time series data must NOT be split randomly. 
    # Use chronological truncation to prevent future data leakage.
    n_samples = len(X_train_south)
    split_idx = int(n_samples * 0.8) # 80% for training, 20% for validation
    
    X_tr_s, X_val_s = X_train_south[:split_idx], X_train_south[split_idx:]
    X_tr_n, X_val_n = X_train_north[:split_idx], X_train_north[split_idx:]
    y_tr, y_val = y_train_all[:split_idx], y_train_all[split_idx:]

    # Construct unified datasets to avoid misalignment
    train_dataset = DualGridDataset3D(X_tr_s, X_tr_n, y_tr)
    val_dataset = DualGridDataset3D(X_val_s, X_val_n, y_val)
    pred_dataset = DualGridDataset3D(X_pred_south, X_pred_north)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Shuffling within training batches is fine
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualTowerGridCNN3D(X_train_south.shape[1:]).to(device)
    
    criterion = torch.nn.SmoothL1Loss(beta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for b_X_s, b_X_n, b_y in train_loader:
            b_X_s, b_X_n, b_y = b_X_s.to(device), b_X_n.to(device), b_y.to(device)
            optimizer.zero_grad()
            outputs = model(b_X_s, b_X_n).squeeze()
            loss = criterion(outputs, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation Loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for b_X_s, b_X_n, b_y in val_loader:
                b_X_s, b_X_n, b_y = b_X_s.to(device), b_X_n.to(device), b_y.to(device)
                outputs = model(b_X_s, b_X_n).squeeze()
                val_loss += criterion(outputs, b_y).item()
                
        avg_val_loss = val_loss / max(1, len(val_loader))
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")

    # Prediction phase
    model.eval()
    predictions = []
    with torch.no_grad():
        for b_X_s, b_X_n in pred_loader:
            b_X_s, b_X_n = b_X_s.to(device), b_X_n.to(device)
            preds = model(b_X_s, b_X_n).squeeze().cpu().numpy()
            predictions.extend(preds if preds.ndim > 0 else [preds])

    return {'predictions': np.array(predictions)}

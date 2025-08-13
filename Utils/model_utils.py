import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import random
import numpy as np
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize(model_params, clean_folder, noisy_folder):
  window = torch.hamming_window(model_params["window_length"], periodic=True)
  clean_audios = list(Path(clean_folder).rglob('*.flac'))
  noisy_audios = list(Path(noisy_folder).rglob('*.flac'))

  noisy_segments_list = []
  clean_segments_list = []

  for i in range(100):
    clean_audio = torchaudio.load(clean_audios[i])[0]
    noisy_audio = torchaudio.load(noisy_audios[i])[0]
    clean_segment, noisy_segment = audio_to_segments(clean_audio, noisy_audio, model_params, window)
    noisy_segments_list.append(noisy_segment.cpu())
    clean_segments_list.append(clean_segment.cpu())
  return noisy_segments_list, clean_segments_list
    
def audio_to_segments(clean_audio, noisy_audio, model_params, window):
    noisy_stft = torch.stft(noisy_audio.squeeze(), n_fft=model_params["fft_length"],
                            hop_length=model_params["window_length"] - model_params["overlap"],
                            window=window, return_complex=True)
    
    clean_stft = torch.stft(clean_audio.squeeze(), n_fft=model_params["fft_length"],
                            hop_length=model_params["window_length"] - model_params["overlap"],
                            window=window, return_complex=True)
    noisy_mag = noisy_stft.abs()
    clean_mag = clean_stft.abs()

    def create_segments(mag, num_segments):
        num_frames = mag.shape[1]
        padded = torch.cat([mag[:, :num_segments-1], mag], dim=1)
        segments = []
        for i in range(num_frames):
            seg = padded[:, i:i+num_segments]
            segments.append(seg.unsqueeze(0))
        return torch.cat(segments, dim=0)
    X = create_segments(noisy_mag, model_params["num_segments"])
    
    return X, clean_mag.T[:noisy_mag.shape[1], :]

def batch_preprocessing(noisy_segments_list, clean_segments_list):
    noisy_segments = torch.cat(noisy_segments_list, dim=0)
    clean_segments = torch.cat(clean_segments_list, dim=0)

    noisy_mean = noisy_segments.mean()
    noisy_std = noisy_segments.std()
    noisy_segments = (noisy_segments - noisy_mean) / noisy_std

    clean_mean = clean_segments.mean()
    clean_std = clean_segments.std()
    clean_segments = (clean_segments - clean_mean) / clean_std

    # Reformat
    X = noisy_segments.unsqueeze(1)  # [batch, 1, freq, segs]
    y = clean_segments  # [batch, freq]

    return X, y, noisy_mean, noisy_std, clean_mean, clean_std

def dataset_split(X, y):
    num_samples = X.shape[0]
    indices = torch.randperm(num_samples)
    train_len = int(0.99 * num_samples)
    train_idx = indices[:train_len]
    val_idx = indices[train_len:]
    batch_size=128

    train_dataset = torch.utils.data.TensorDataset(X[train_idx], y[train_idx])
    val_dataset = torch.utils.data.TensorDataset(X[val_idx], y[val_idx])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader

def train_model(model, epochs, model_params, train_loader, val_loader):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=model_params["lr"])

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.6f} - Val Loss: {avg_val_loss:.6f}")

    return model

class DenoiseNet(nn.Module):
    def __init__(self, num_features, num_segments):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(num_features*num_segments, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, num_features)
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
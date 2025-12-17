# ===============================
# SisFall Data Loader + 1D CNN Model
# FSOS Project (IMU-based Fall Detection)
# ===============================

"""
This file contains:
1. SisFall dataset loader (raw TXT -> windows)
2. Label extraction (Fall vs ADL)
3. PyTorch Dataset & DataLoader
4. Lightweight 1D CNN architecture (mobile-friendly)

Designed for:
- Accelerometer + Gyroscope (6 channels)
- Phone-side inference later (TFLite/ONNX)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# -------------------------------
# Configuration
# -------------------------------

SISFALL_ROOT = "/home/sud/fsos_ring/datasets/SisFall_dataset"
WINDOW_SIZE = 200        # 200 samples = 1 sec at 200 Hz
WINDOW_STRIDE = 100      # 50% overlap
NUM_CHANNELS = 6         # Ax Ay Az Gx Gy Gz

# -------------------------------
# Utility: detect fall or ADL
# -------------------------------

def is_fall_file(filename: str) -> int:
    """
    SisFall naming convention:
    F = Fall
    D = ADL (Daily Activity)
    """
    return 1 if filename.startswith('F') else 0

# -------------------------------
# Load a single SisFall file
# -------------------------------

def load_sisfall_txt(path):
    """
    Each row format (SisFall):
    ax ay az gx gy gz (+ 3 extra columns we ignore)
    Rows end with semicolon, values comma-separated
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # Strip semicolons and parse, filtering empty lines
    cleaned_lines = [line.rstrip(';\n').strip() for line in lines if line.strip()]
    data = np.array([list(map(float, line.split(','))) for line in cleaned_lines])
    
    # Keep only first 6 channels (ax, ay, az, gx, gy, gz)
    data = data[:, :NUM_CHANNELS]
    return data.astype(np.float32)

# -------------------------------
# Sliding window segmentation
# -------------------------------

def create_windows(signal, window_size, stride):
    windows = []
    for start in range(0, len(signal) - window_size, stride):
        window = signal[start:start + window_size]
        windows.append(window)
    return np.array(windows)

# -------------------------------
# PyTorch Dataset
# -------------------------------

class SisFallDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.labels = []

        for subject in os.listdir(root_dir):
            subject_path = os.path.join(root_dir, subject)
            if not os.path.isdir(subject_path):
                continue

            for file in os.listdir(subject_path):
                if not file.endswith('.txt'):
                    continue

                label = is_fall_file(file)
                full_path = os.path.join(subject_path, file)

                signal = load_sisfall_txt(full_path)

                # Normalize (z-score per channel)
                signal = (signal - signal.mean(axis=0)) / (signal.std(axis=0) + 1e-6)

                windows = create_windows(signal, WINDOW_SIZE, WINDOW_STRIDE)

                self.samples.extend(windows)
                self.labels.extend([label] * len(windows))

        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx]).transpose(0, 1)  # (C, T)
        y = torch.tensor(self.labels[idx]).long()
        return x, y

# -------------------------------
# Lightweight 1D CNN Model
# -------------------------------

class FallDetectionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(NUM_CHANNELS, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # Fall / Non-fall
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classifier(x)

# -------------------------------
# Example usage
# -------------------------------

if __name__ == '__main__':
    dataset = SisFallDataset(SISFALL_ROOT)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = FallDetectionCNN()

    x, y = next(iter(loader))
    out = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", out.shape)

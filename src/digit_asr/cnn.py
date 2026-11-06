"""Optional CPU MFCC baseline; importing the default package never imports torch."""

import numpy as np
import torch
from torch import nn


class CNNClassifier:
    def __init__(self, seed=42, epochs=100, batch_size=32, learning_rate=0.003):
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def fit(self, x, y):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 3 or len(x) != len(y) or len(x) == 0:
            raise ValueError("Expected nonempty (samples, MFCC channels, frames) and labels")
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed)
                self.mean_ = x.mean(axis=(0, 2), keepdims=True)
                self.scale_ = np.maximum(x.std(axis=(0, 2), keepdims=True), 1e-6)
                inputs = torch.from_numpy((x - self.mean_) / self.scale_)
                targets = torch.as_tensor(np.asarray(y), dtype=torch.long)
                self.network_ = nn.Sequential(
                    nn.Conv1d(x.shape[1], 16, 5, padding=2),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(16, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(4),
                    nn.Flatten(),
                    nn.Linear(128, 10),
                )
                optimizer = torch.optim.Adam(
                    self.network_.parameters(), lr=self.learning_rate, weight_decay=1e-4
                )
                self.network_.train()
                for _ in range(self.epochs):
                    for batch in torch.randperm(len(inputs)).split(self.batch_size):
                        optimizer.zero_grad()
                        loss = nn.functional.cross_entropy(
                            self.network_(inputs[batch]), targets[batch]
                        )
                        loss.backward()
                        optimizer.step()
                self.network_.eval()
        finally:
            torch.set_num_threads(previous_threads)
        return self

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.network_.parameters())

    def predict(self, x):
        inputs = torch.from_numpy((np.asarray(x, dtype=np.float32) - self.mean_) / self.scale_)
        with torch.inference_mode():
            return self.network_(inputs).argmax(dim=1).numpy()

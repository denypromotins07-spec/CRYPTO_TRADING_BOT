"""
TimeGAN Implementation for Synthetic Tick Generation

Lightweight Time-Series Generative Adversarial Network optimized for:
- Generating realistic crypto tick sequences
- Preserving temporal dynamics and autocorrelation
- Maintaining fat-tailed return distributions
- Avoiding mode collapse

Memory-efficient architecture for 8GB RAM constraint.
Uses PyTorch with careful memory management.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TimeGANConfig:
    """Configuration for TimeGAN architecture."""
    sequence_length: int = 100        # Length of generated sequences
    num_features: int = 5             # Features: price, volume, spread, etc.
    embedding_dim: int = 32           # Embedding space dimension
    hidden_dim: int = 64              # LSTM hidden dimension
    num_layers: int = 2               # Number of LSTM layers
    learning_rate: float = 0.001      # Adam learning rate
    beta1: float = 0.5                # Adam beta1
    beta2: float = 0.999              # Adam beta2
    lambda_reconstruction: float = 100.0  # Reconstruction loss weight
    lambda_supervised: float = 100.0      # Supervised loss weight
    gradient_clip: float = 5.0        # Gradient clipping value
    dropout: float = 0.2              # Dropout rate
    
    @classmethod
    def crypto_ticks(cls) -> TimeGANConfig:
        """Config optimized for crypto tick data."""
        return cls(
            sequence_length=50,
            num_features=6,  # mid_price, spread, bid_vol, ask_vol, order_imbalance, volatility
            embedding_dim=24,
            hidden_dim=48,
            num_layers=2,
            learning_rate=0.0005,
        )


class EmbeddingNetwork(nn.Module):
    """Maps input data to latent space and back."""
    
    def __init__(self, config: TimeGANConfig):
        super().__init__()
        self.config = config
        
        # Encoder: feature space -> embedding space
        self.encoder = nn.Sequential(
            nn.Linear(config.num_features, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )
        
        # Decoder: embedding space -> feature space
        self.decoder = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_features),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent space and decode back."""
        embedded = self.encoder(x)
        reconstructed = self.decoder(embedded)
        return embedded, reconstructed


class Generator(nn.Module):
    """Generates synthetic sequences from random noise."""
    
    def __init__(self, config: TimeGANConfig):
        super().__init__()
        self.config = config
        self.sequence_length = config.sequence_length
        
        # LSTM-based generator
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
        )
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
            nn.Tanh(),  # Bound outputs
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
    
    def forward(
        self, 
        noise: torch.Tensor,
        initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate synthetic sequence.
        
        Args:
            noise: Random noise tensor [batch, seq_len, embedding_dim]
            initial_state: Optional initial hidden state
            
        Returns:
            Generated sequence and final hidden state
        """
        lstm_out, hidden_state = self.lstm(noise, initial_state)
        output = self.fc(lstm_out)
        return output, hidden_state


class Discriminator(nn.Module):
    """Distinguishes real from synthetic sequences."""
    
    def __init__(self, config: TimeGANConfig):
        super().__init__()
        self.config = config
        
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify sequence as real or fake."""
        lstm_out, _ = self.lstm(x)
        # Use last time step for classification
        output = self.classifier(lstm_out[:, -1, :])
        return output.squeeze(-1)


class Supervisor(nn.Module):
    """Helps generator learn temporal dynamics."""
    
    def __init__(self, config: TimeGANConfig):
        super().__init__()
        self.config = config
        
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
        )
        
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
            nn.Tanh(),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict next embedding given current sequence."""
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output


class TimeGAN:
    """
    Complete TimeGAN implementation for synthetic tick generation.
    
    Training procedure:
    1. Pre-train embedding network (reconstruction)
    2. Pre-train supervisor (temporal dynamics)
    3. Joint training of generator and discriminator
    """
    
    def __init__(self, config: Optional[TimeGANConfig] = None, device: Optional[str] = None):
        self.config = config or TimeGANConfig.crypto_ticks()
        
        # Device selection (CPU for 8GB RAM constraint)
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"TimeGAN initialized on {self.device}")
        
        # Initialize networks
        self.embedder = EmbeddingNetwork(self.config).to(self.device)
        self.generator = Generator(self.config).to(self.device)
        self.discriminator = Discriminator(self.config).to(self.device)
        self.supervisor = Supervisor(self.config).to(self.device)
        
        # Optimizers
        self.opt_embedder = optim.Adam(self.embedder.parameters(), lr=self.config.learning_rate)
        self.opt_generator = optim.Adam(self.generator.parameters(), lr=self.config.learning_rate)
        self.opt_discriminator = optim.Adam(self.discriminator.parameters(), lr=self.config.learning_rate)
        self.opt_supervisor = optim.Adam(self.supervisor.parameters(), lr=self.config.learning_rate)
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
        # Training state
        self.training_history: Dict[str, List[float]] = {
            'loss_D': [],
            'loss_G': [],
            'loss_E': [],
            'loss_S': [],
        }
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to [-1, 1] range."""
        min_val = data.min(axis=(0, 1), keepdims=True)
        max_val = data.max(axis=(0, 1), keepdims=True)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1  # Avoid division by zero
        normalized = 2 * (data - min_val) / range_val - 1
        return normalized
    
    def _denormalize_data(self, normalized: np.ndarray, original_min: np.ndarray, 
                          original_max: np.ndarray) -> np.ndarray:
        """Denormalize data from [-1, 1] to original range."""
        range_val = original_max - original_min
        denormalized = (normalized + 1) / 2 * range_val + original_min
        return denormalized
    
    def pretrain_embedder(self, data: np.ndarray, epochs: int = 100, batch_size: int = 64) -> List[float]:
        """Pre-train embedding network for reconstruction."""
        # Normalize data
        self.data_min = data.min(axis=(0, 1), keepdims=True)
        self.data_max = data.max(axis=(0, 1), keepdims=True)
        normalized = self._normalize_data(data)
        
        data_tensor = torch.FloatTensor(normalized).to(self.device)
        dataset = torch.utils.data.TensorDataset(data_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in loader:
                x = batch[0]
                
                self.opt_embedder.zero_grad()
                _, reconstructed = self.embedder(x)
                loss = self.mse_loss(reconstructed, x)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.embedder.parameters(), self.config.gradient_clip)
                self.opt_embedder.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)
            
            if epoch % 10 == 0:
                logger.debug(f"Embedder epoch {epoch}: loss={avg_loss:.6f}")
        
        return losses
    
    def pretrain_supervisor(self, data: np.ndarray, epochs: int = 100, batch_size: int = 64) -> List[float]:
        """Pre-train supervisor for temporal dynamics."""
        normalized = self._normalize_data(data)
        data_tensor = torch.FloatTensor(normalized).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            embeddings, _ = self.embedder(data_tensor)
        
        dataset = torch.utils.data.TensorDataset(embeddings)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in loader:
                emb = batch[0]
                
                # Input: all but last timestep
                # Target: all but first timestep
                emb_input = emb[:, :-1, :]
                emb_target = emb[:, 1:, :]
                
                self.opt_supervisor.zero_grad()
                prediction = self.supervisor(emb_input)
                loss = self.mse_loss(prediction, emb_target)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.supervisor.parameters(), self.config.gradient_clip)
                self.opt_supervisor.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)
        
        return losses
    
    def train_adversarial(self, data: np.ndarray, epochs: int = 100, batch_size: int = 64) -> Dict[str, List[float]]:
        """Joint adversarial training of generator and discriminator."""
        normalized = self._normalize_data(data)
        data_tensor = torch.FloatTensor(normalized).to(self.device)
        
        with torch.no_grad():
            embeddings, _ = self.embedder(data_tensor)
        
        dataset = torch.utils.data.TensorDataset(embeddings)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            loss_D_total = 0.0
            loss_G_total = 0.0
            
            for batch in loader:
                emb_real = batch[0]
                batch_size_current = emb_real.size(0)
                
                # --- Train Discriminator ---
                self.opt_discriminator.zero_grad()
                
                # Real samples
                d_real = self.discriminator(emb_real)
                loss_d_real = self.bce_loss(d_real, torch.ones_like(d_real))
                
                # Fake samples
                noise = torch.randn(batch_size_current, self.config.sequence_length, 
                                   self.config.embedding_dim).to(self.device)
                emb_fake, _ = self.generator(noise)
                d_fake = self.discriminator(emb_fake.detach())
                loss_d_fake = self.bce_loss(d_fake, torch.zeros_like(d_fake))
                
                loss_D = (loss_d_real + loss_d_fake) / 2
                loss_D.backward()
                
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config.gradient_clip)
                self.opt_discriminator.step()
                
                # --- Train Generator ---
                self.opt_generator.zero_grad()
                
                # Generator wants to fool discriminator
                emb_fake, _ = self.generator(noise)
                d_fake = self.discriminator(emb_fake)
                loss_G_adv = self.bce_loss(d_fake, torch.ones_like(d_fake))
                
                # Supervised loss (match temporal dynamics)
                emb_fake_sup = self.supervisor(emb_fake[:, :-1, :])
                loss_G_sup = self.mse_loss(emb_fake_sup, emb_fake[:, 1:, :])
                
                # Reconstruction loss
                _, reconstructed = self.embedder(self.decoder_output(emb_fake))
                loss_G_rec = self.mse_loss(reconstructed, self.decoder_output(emb_fake))
                
                loss_G = (loss_G_adv + 
                         self.config.lambda_supervised * loss_G_sup + 
                         self.config.lambda_reconstruction * loss_G_rec)
                loss_G.backward()
                
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.config.gradient_clip)
                self.opt_generator.step()
                
                loss_D_total += loss_D.item()
                loss_G_total += loss_G.item()
            
            avg_loss_D = loss_D_total / len(loader)
            avg_loss_G = loss_G_total / len(loader)
            
            self.training_history['loss_D'].append(avg_loss_D)
            self.training_history['loss_G'].append(avg_loss_G)
            
            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}: D_loss={avg_loss_D:.4f}, G_loss={avg_loss_G:.4f}")
        
        return self.training_history
    
    def decoder_output(self, embedding: torch.Tensor) -> torch.Tensor:
        """Helper to get feature space from embedding."""
        # Flatten for decoder
        batch_size, seq_len, emb_dim = embedding.shape
        flat_emb = embedding.view(-1, emb_dim)
        decoded = self.embedder.decoder(flat_emb)
        return decoded.view(batch_size, seq_len, -1)
    
    def generate(self, num_sequences: int) -> np.ndarray:
        """Generate synthetic tick sequences."""
        self.generator.eval()
        self.embedder.eval()
        
        with torch.no_grad():
            noise = torch.randn(num_sequences, self.config.sequence_length, 
                               self.config.embedding_dim).to(self.device)
            emb_generated, _ = self.generator(noise)
            
            # Decode to feature space
            batch_size, seq_len, emb_dim = emb_generated.shape
            flat_emb = emb_generated.view(-1, emb_dim)
            decoded = self.embedder.decoder(flat_emb)
            generated = decoded.view(batch_size, seq_len, -1)
            
            # Denormalize
            generated_np = generated.cpu().numpy()
            denormalized = self._denormalize_data(generated_np, self.data_min, self.data_max)
        
        return denormalized
    
    def check_mode_collapse(self, generated: np.ndarray, threshold: float = 0.5) -> bool:
        """Detect mode collapse by checking diversity."""
        if generated.shape[0] < 10:
            return False
        
        # Calculate pairwise distances
        flat_gen = generated.reshape(generated.shape[0], -1)
        
        # Sample pairs for efficiency
        n_samples = min(100, flat_gen.shape[0])
        indices = np.random.choice(flat_gen.shape[0], n_samples, replace=False)
        sampled = flat_gen[indices]
        
        distances = []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.linalg.norm(sampled[i] - sampled[j])
                distances.append(dist)
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        # Mode collapse if low diversity (small distances)
        collapsed = mean_dist < threshold
        
        if collapsed:
            logger.warning(f"Potential mode collapse detected: mean_distance={mean_dist:.4f}")
        else:
            logger.info(f"Diversity check passed: mean_distance={mean_dist:.4f}")
        
        return collapsed
    
    def save_model(self, path: str) -> None:
        """Save model checkpoints."""
        checkpoint = {
            'embedder': self.embedder.state_dict(),
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'supervisor': self.supervisor.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model checkpoints."""
        checkpoint = torch.load(path, map_location=self.device)
        self.embedder.load_state_dict(checkpoint['embedder'])
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.supervisor.load_state_dict(checkpoint['supervisor'])
        self.training_history = checkpoint.get('training_history', self.training_history)
        logger.info(f"Model loaded from {path}")


# Example usage
if __name__ == "__main__":
    # Create synthetic training data
    np.random.seed(42)
    n_samples = 1000
    seq_len = 50
    n_features = 6
    
    # Simulate realistic tick data
    time = np.arange(seq_len)
    data = np.zeros((n_samples, seq_len, n_features))
    
    for i in range(n_samples):
        # Random walk with drift for price
        price = np.cumsum(np.random.randn(seq_len) * 0.01) + time * 0.001
        # Volume clustering
        volume = np.abs(np.random.randn(seq_len)) * np.exp(-time / 20) + 1
        # Spread mean-reverting
        spread = 0.001 + 0.0005 * np.sin(time / 10) + np.random.randn(seq_len) * 0.0001
        
        data[i, :, 0] = price
        data[i, :, 1] = spread
        data[i, :, 2] = volume
        data[i, :, 3] = volume * np.random.rand(seq_len)  # bid vol
        data[i, :, 4] = np.random.randn(seq_len)  # order imbalance
        data[i, :, 5] = np.abs(np.random.randn(seq_len))  # volatility
    
    # Initialize and train TimeGAN
    gan = TimeGAN(TimeGANConfig.crypto_ticks())
    
    print("Pre-training embedder...")
    embedder_losses = gan.pretrain_embedder(data, epochs=50, batch_size=32)
    
    print("Pre-training supervisor...")
    supervisor_losses = gan.pretrain_supervisor(data, epochs=50, batch_size=32)
    
    print("Adversarial training...")
    history = gan.train_adversarial(data, epochs=100, batch_size=32)
    
    # Generate synthetic data
    print("Generating synthetic sequences...")
    synthetic = gan.generate(num_sequences=100)
    print(f"Generated shape: {synthetic.shape}")
    
    # Check for mode collapse
    collapsed = gan.check_mode_collapse(synthetic)
    print(f"Mode collapse detected: {collapsed}")

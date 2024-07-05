import torch
import torch.nn as nn
from torch import Tensor
import math

class FourierEmbedding(nn.Module):
    def __init__(self, num_frequencies: int, max_freq: float = 10):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.frequencies = nn.Parameter(torch.linspace(1.0, max_freq, num_frequencies), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch_size, seq_len, 2)
        batch_size, seq_len, _ = x.shape
        
        # Compute sin and cos for each frequency
        x = x.unsqueeze(-1) * self.frequencies.view(1, 1, 1, -1) * 2 * math.pi
        sin_x = torch.sin(x)
        cos_x = torch.cos(x)
        
        # Concatenate sin and cos
        embeddings = torch.cat([sin_x, cos_x], dim=-1)
        
        # Flatten the last two dimensions
        return embeddings.view(batch_size, seq_len, -1)


class TransformerNet(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 emb_size: int,
                 num_head: int,
                 num_classes: int,
                 num_fourier_features: int = 10):
        super().__init__()
        
        # Initialise the Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size,
                                                   nhead=num_head,
                                                   dim_feedforward=emb_size * 4,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Embed adc and position
        self.adc_emb = nn.Linear(2, emb_size // 2)
        self.fourier_embedding = FourierEmbedding(num_fourier_features)
        fourier_feature_size = num_fourier_features * 4  # 2 (sin, cos) * 2 (x, y)
        self.pos_encoding = nn.Linear(fourier_feature_size, emb_size // 2)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, num_classes)
        )


    def forward(self, src: Tensor):
        # src shape: (batch_size, seq_len, 4)
        batch_size, seq_len, _ = src.shape
        
        # Create padding mask
        padding_mask = (src[:,:,0] == -1000)  
        
        # Split source into positions and ADC values
        positions, adcs = src[:, :, :2], src[:, :, 2:]
        
        # Embed the ADC values
        adc_embed = self.adc_emb(adcs)
        fourier = self.fourier_embedding(positions)
        pos_embed = self.pos_encoding(fourier)
        
        # Combine ADC embeddings and positional encoding
        x = torch.cat((adc_embed, pos_embed), dim=-1)
        
        # Apply the Transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # Global average pooling (excluding padded values)
        mask = ~padding_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1)
        
        # Classify
        output = self.classifier(x)
        
        return output



# Example usage
if __name__ == "__main__":
    model = TransformerNet(num_encoder_layers=3, emb_size=128, num_head=2, num_classes=1)
    
    # Example input
    batch_size, seq_len, num_features = 32, 190, 4
    input_data = torch.randn(batch_size, seq_len, num_features)
    
    # Forward pass
    output = model(input_data)
    print(f"Output shape: {output.shape}")  # Should be (32,)
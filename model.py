import torch
import torch.nn as nn
from torch import Tensor

class TransformerNet(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 emb_size: int,
                 num_head: int,
                 num_classes: int):
        super().__init__()
        
        # Initialise the Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size,
                                                   nhead=num_head,
                                                   dim_feedforward=emb_size * 4,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Embed adc and position
        self.adc_emb = nn.Linear(2, emb_size // 2)
        self.pos_encoding = nn.Linear(2, emb_size // 2)
        
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
        pos_embed = self.pos_encoding(positions)
        
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
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict

class LinearEmbedding(nn.Module):
    def __init__(self, num_patches, input_feature_dim, output_dim, isRegion=False):
        super(LinearEmbedding, self).__init__()
        self.num_patches = num_patches
        self.output_dim = output_dim    # D = D_e or D_r
        self.isRegion = isRegion

        self.class_token = nn.Parameter(torch.randn(1, output_dim), requires_grad=True)
        # self.class_token = nn.Parameter(torch.randn(1, output_dim))  # (1, output_dim)

        self.projection = nn.Linear(in_features=input_feature_dim, out_features=output_dim)
        
        self.position_embedding = nn.Embedding(num_embeddings=num_patches + 1, embedding_dim=output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, patch):   # patch: (B, num_patches, input_feature_dim)
        batch_size = patch.shape[0]
        class_token = self.class_token.expand(batch_size, -1) # (B, output_dim)
        class_token = class_token.unsqueeze(1)  # (B, 1, output_dim)
        patches_embed = self.projection(patch)  # (B, num_patches, output_dim)
        if self.isRegion is True:  # for region-level spatial learning
            patches_embed = self.dropout(patches_embed) # use 0.1 dropout 
        embedded_sequence = torch.cat([class_token, patches_embed], dim=1)  # (B, num_patches + 1, output_dim)
        
        positions = torch.arange(start=0, end=self.num_patches+1, step=1, device=patch.device) # positions: (num_patches + 1,)
        positions_embed = self.position_embedding(positions) # positions_embed: (num_patches + 1, output_dim)

        encoded = embedded_sequence + positions_embed  # (B, num_patches + 1, output_dim)
        
        # encoded: (B, num_patches + 1, D_e) or (B, num_patches + 1, D_r)
        return encoded



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.4, activation='relu'):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f'activation function {activation} is not supported')

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        return x



class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, heads, dropout=0.4, D_h=64, k=16):
        super(TransformerEncoderLayer, self).__init__()
        self.ln1 = nn.LayerNorm(input_dim, eps=1e-6)
        self.msa = nn.MultiheadAttention(embed_dim=input_dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(input_dim, eps=1e-6)  
        self.mlp = MLP(input_dim=input_dim, hidden_dim=input_dim*4, output_dim=input_dim, dropout_rate=dropout, activation='gelu')

    def forward(self, x):
        x1 = self.ln1(x)
        attn_output, _ = self.msa(x1, x1, x1)
        y = x + attn_output
        z = self.mlp(self.ln2(y)) + y
        return z


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, heads, num_layers=2, dropout=0.4, D_h=64, k=16):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(input_dim=input_dim, heads=heads, dropout=dropout) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



class HSLT(nn.Module):
    def __init__(self, num_classes, debug=False):
        super(HSLT, self).__init__()
        self.num_classes = num_classes
        self.debug = debug
        self.d_e = 8  
        self.d_r = 16

        self.electrode_level = nn.ModuleDict()
        self.adapters = nn.ModuleDict() 

        unique_processed_seq_lens = set()
        region_electrode_mapping = {
            "PF": 4, "F": 5, "LT": 3, "C": 5, "RT": 3, 
            "LP": 3, "P": 3, "RP": 3, "O": 3
        }

        for region_name, num_electrodes in region_electrode_mapping.items():
            current_processed_seq_len = num_electrodes + 1
            unique_processed_seq_lens.add(current_processed_seq_len)
            self.electrode_level[region_name] = nn.Sequential(
                LinearEmbedding(
                    num_patches=num_electrodes, 
                    input_feature_dim=5, 
                    output_dim=self.d_e, 
                    isRegion=False
                ),
                TransformerEncoder(
                    input_dim=self.d_e, 
                    heads=2),
            )
        
        for seq_len in unique_processed_seq_lens:
            if seq_len != 4:
                self.adapters[str(seq_len)] = nn.Linear(seq_len, 4)

        self.region_level = nn.Sequential(
            LinearEmbedding(
                num_patches=9, 
                input_feature_dim=4 * self.d_e, 
                output_dim=self.d_r,
                isRegion=True
            ), 
            TransformerEncoder(
                input_dim=self.d_r, 
                heads=2), 
        )
        
        self.predictor = nn.Linear(in_features=self.d_r, out_features=num_classes)
    
    def forward(self, x: Dict[str, torch.Tensor]):
        if self.debug:
            print("\ninput shape:")
            for region_name, region_data in x.items():
                print(f"shape of {region_name}: {region_data.shape}") # (B, num_electrodes, 5)

        processed_electrode_data = {}
        for region_name, region_data in x.items(): 
            processed_data = self.electrode_level[region_name](region_data)
            current_seq_len = processed_data.shape[1]
            if current_seq_len != 4:
                processed_data = processed_data.permute(0, 2, 1) 
                adapter_key = str(current_seq_len)
                if adapter_key in self.adapters:
                    processed_data = self.adapters[adapter_key](processed_data)
                processed_data = processed_data.permute(0, 2, 1) 
            
            processed_data = processed_data.unsqueeze(1) 
            processed_electrode_data[region_name] = processed_data

        if self.debug :
            print("\nshape after electrode-level processing and length adaptation:")
            for region_name, region_data in processed_electrode_data.items():
                print(f"shape of {region_name}: {region_data.shape}") 

        X_R = torch.cat(list(processed_electrode_data.values()), dim=1)
        if self.debug:
            print(f"\nshape of X_R (before reshape): {X_R.shape}")

        X_R = X_R.reshape(X_R.shape[0], 9, -1)
        z = self.region_level(X_R) 
        z_0 = z[:, 0, :]
        if self.debug:
            print(f"\nshape of z_0: {z_0.shape}")

        y_pred = self.predictor(z_0) # (B, num_classes)
        return y_pred


if __name__ == "__main__":
    x = {
        "PF": torch.randn(4, 4, 5), 
        "F": torch.randn(4, 5, 5),  
        "LT": torch.randn(4, 3, 5), 
        "C": torch.randn(4, 5, 5),
        "RT": torch.randn(4, 3, 5),
        "LP": torch.randn(4, 3, 5),
        "P": torch.randn(4, 3, 5),
        "RP": torch.randn(4, 3, 5),
        "O": torch.randn(4, 3, 5),
    }

    hslt = HSLT(num_classes=2, debug=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hslt.to(device)
    
    for region_name in x:
        x[region_name] = x[region_name].to(device)

    y = hslt(x, device=device)
    print("Final output shape:", y.shape, "\nFinal output:", y)
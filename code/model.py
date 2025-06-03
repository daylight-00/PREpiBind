import torch.nn.functional as F
import torch.nn as nn
import torch

#%% PLM BLOCKS
class _ffn_residual_self_attn_block(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(_ffn_residual_self_attn_block, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        attn_output, _ = self.multihead_attn(x, x, x, key_padding_mask=padding_mask)
        x = self.layer_norm1(x + self.dropout(attn_output))
        
        ffn_output = self.ffn(x)    
        x = self.layer_norm2(x + self.dropout(ffn_output))
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim)
        return x

class simple_self_attn(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, n_blocks=1):
        super(simple_self_attn, self).__init__()
        self.blocks = nn.ModuleList([
            _ffn_residual_self_attn_block(embed_dim, num_heads, dropout) for _ in range(n_blocks)
        ])
        
    def forward(self, x, padding_mask=None):
        for block in self.blocks:
            x = block(x, padding_mask)
        return x

#%% PLM MODELS
class plm_cat_mean_inf(nn.Module):
    def __init__(
            self, 
            hla_dim=384, epi_dim=384, 
            dropout=0.1, hla_blocks=2, epi_blocks=2, con_blocks=1, head_div=64
        ):
        super(plm_cat_mean_inf, self).__init__()

        nhead = hla_dim // head_div
        self.epi_self_attn = simple_self_attn(embed_dim=epi_dim, num_heads=nhead, n_blocks=epi_blocks, dropout=dropout)
        self.hla_self_attn = simple_self_attn(embed_dim=hla_dim, num_heads=nhead, n_blocks=hla_blocks, dropout=dropout)
        
        concat_dim = hla_dim
        nhead = concat_dim // head_div
        self.self_attn = simple_self_attn(embed_dim=concat_dim, num_heads=nhead, n_blocks=con_blocks, dropout=dropout)
        
        flat_dim = concat_dim
        self.output_layer = nn.Sequential(
            nn.Linear(flat_dim, flat_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(flat_dim//2, 1)
        )

    def forward(self, x_hla, x_epi, mask_hla=None, mask_epi=None):
        x_hla = self.hla_self_attn(x_hla, padding_mask=mask_hla)
        x_epi = self.epi_self_attn(x_epi, padding_mask=None)

        mask = torch.cat((mask_hla, mask_epi), dim=-1)
        x = torch.cat((x_hla, x_epi), dim=1)
        x = self.self_attn(x, padding_mask=mask)

        mask = (~mask).unsqueeze(-1)
        x = x*mask
        x = x.sum(dim=1) / (mask.sum(dim=1))
        x = self.output_layer(x)
        return x

class UnifiedModel(nn.Module):
    def __init__(self, model_esm, model):
        super().__init__()
        self.model_esm = model_esm
        self.model = model

    def forward(self, x_hla, x_epi, mask_hla, mask_epi):
        x_epi_emb = self.model_esm(x_epi).embeddings[:, 1:-1, :]
        y_pred = self.model(x_hla, x_epi_emb, mask_hla, mask_epi)
        return y_pred

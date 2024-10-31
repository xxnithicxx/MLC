import torch
import torch.nn as nn
from torch.nn.functional import relu
from src.config import CONFIG

# Source: https://github.com/Alibaba-MIIL/ML_Decoder/blob/main/src_files/ml_decoder/ml_decoder.py

def add_ml_decoder_head(model, num_classes, num_of_groups=-1, decoder_embedding=768, zsl=0):
    if num_classes == -1:
        num_classes = model.num_classes
        
    # Set num_features manually for ResNet-50 (output of the last conv layer)
    num_features = 2048
    model.fc = MLDecoder(num_classes=num_classes, initial_num_features=num_features,
                         num_of_groups=num_of_groups, decoder_embedding=decoder_embedding, zsl=zsl)

    return model

class GroupFC:
    def __init__(self, embed_len_decoder):
        self.embed_len_decoder = embed_len_decoder

    def __call__(self, h, duplicate_pooling, out_extrap):
        for i in range(h.shape[1]):
            h_i = h[:, i, :]
            w_i = duplicate_pooling[i, :, :] if len(duplicate_pooling.shape) == 3 else duplicate_pooling
            out_extrap[:, i, :] = torch.matmul(h_i, w_i)

class MLDecoder(nn.Module):
    def __init__(self, num_classes, decoder_embedding=768, initial_num_features=2048, num_of_groups=-1, zsl=0):
        super(MLDecoder, self).__init__()
        embed_len_decoder = 100 if num_of_groups < 0 else num_of_groups
        embed_len_decoder = min(embed_len_decoder, num_classes)

        self.embed_standart = nn.Linear(initial_num_features, decoder_embedding)
        self.query_embed = nn.Embedding(embed_len_decoder, decoder_embedding) if not zsl else None
        if self.query_embed:
            self.query_embed.requires_grad_(False)

        layer_decode = nn.TransformerDecoderLayer(
            d_model=decoder_embedding, nhead=8, dim_feedforward=2048, dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(layer_decode, num_layers=1)
        self.decoder.duplicate_pooling = nn.Parameter(
            torch.Tensor(embed_len_decoder, decoder_embedding, num_classes // embed_len_decoder + 1)
        )
        self.decoder.duplicate_pooling_bias = nn.Parameter(torch.Tensor(num_classes))
        self.decoder.group_fc = GroupFC(embed_len_decoder)

        torch.nn.init.xavier_normal_(self.decoder.duplicate_pooling)
        torch.nn.init.constant_(self.decoder.duplicate_pooling_bias, 0)

    def forward(self, x):
        embedding_spatial = x.flatten(2).transpose(1, 2) if len(x.shape) == 4 else x
        embedding_spatial = relu(self.embed_standart(embedding_spatial))

        bs = embedding_spatial.shape[0]
        query_embed = self.query_embed.weight if self.query_embed else None
        tgt = query_embed.unsqueeze(1).expand(-1, bs, -1)

        if self.training:
            h = self.decoder(tgt, embedding_spatial.unsqueeze(0)).transpose(0, 1)
        else:
            h = self.decoder(tgt, embedding_spatial.unsqueeze(0).transpose(0, 1)).transpose(0, 1)

        out_extrap = torch.zeros(
            h.shape[0], h.shape[1], self.decoder.duplicate_pooling.shape[2], device=h.device, dtype=h.dtype
        )
        self.decoder.group_fc(h, self.decoder.duplicate_pooling, out_extrap)

        logits = out_extrap.flatten(1)[:, :self.decoder.duplicate_pooling_bias.shape[0]]
        logits += self.decoder.duplicate_pooling_bias
        return logits

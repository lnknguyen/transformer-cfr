
import sys
sys.path.append("../")

import torch
import torch.nn as nn
import torch.nn.functional as F


'''
3-head architecture: treatment and potential outcomes y_0 y_1
'''
class MIMIC_Transformer(nn.Module):
    
    def __init__(
        self, 
        cfg,
        vocab_size: int,
        padding_idx = 0,
        batch_first:bool=True):
        
        super().__init__()
        
        self.gender_embed = nn.Embedding(num_embeddings=3, embedding_dim=1)

        self.diag_embed = nn.Embedding(num_embeddings=vocab_size,
                                        embedding_dim= cfg.MODEL.EMBEDDING_DIM,
                                        padding_idx=padding_idx)

        self.embed_sum = EmbeddingAdder()

        encoder_layer = nn.TransformerEncoderLayer(d_model = cfg.MODEL.EMBEDDING_DIM + 3, nhead= 4)

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = cfg.MODEL.ENCODER_NUM_LAYER,
        )

        self.y0_fc = nn.Sequential(
            nn.Linear(cfg.MODEL.EMBEDDING_DIM + 3, cfg.MODEL.FC_HIDDEN_SIZE),
            nn.BatchNorm1d(cfg.MODEL.FC_HIDDEN_SIZE),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(cfg.MODEL.FC_HIDDEN_SIZE, 1)
        )

        self.y1_fc = nn.Sequential(
            nn.Linear(cfg.MODEL.EMBEDDING_DIM + 3, cfg.MODEL.FC_HIDDEN_SIZE),
            nn.BatchNorm1d(cfg.MODEL.FC_HIDDEN_SIZE),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(cfg.MODEL.FC_HIDDEN_SIZE, 1)
        )
        
        self.tr_fc = nn.Linear(cfg.MODEL.EMBEDDING_DIM + 3, 1) # treatment
        self.bce_loss_fn = nn.BCEWithLogitsLoss() #
        
    def forward(
        self, 
        x_diag: torch.LongTensor,
        x_lengths: torch.LongTensor,
        x_gender: torch.LongTensor,
        yf: torch.FloatTensor,
        t: torch.FloatTensor,
        x_timedelta: torch.FloatTensor = None,
        x_age: torch.FloatTensor = None,
        ) -> torch.Tensor:

        batch_size, max_seq_length, _ = x_diag.shape
        
         #### Embeddings
        x_embed = self.diag_embed(x_diag)

        x_gender = (
            self.gender_embed(x_gender)
            .unsqueeze(1)
            .expand(batch_size, max_seq_length, -1)
        )

        x_age = x_age.unsqueeze(1).expand(batch_size, max_seq_length, -1)
        
        x_timedelta = x_timedelta.unsqueeze(1).expand(
            batch_size, max_seq_length, -1
        )

        # Summing embedded values
        x_embed_sum = self.embed_sum(x_embed)
        x_embed_sum = torch.cat([x_embed_sum, x_gender, x_age, x_timedelta],dim=2)
        
        x_trans = self.transformer(x_embed_sum)
        x_seq= x_trans.mean(1)
        
        t_logits = self.tr_fc(x_seq).view(-1)

        t_loss = self.bce_loss_fn(t_logits, t)
        factual_loss = self.bce_loss_fn(d_preds, yf)

        loss =  factual_loss + t_loss
        
        return (
            loss,
            d_preds,
            t_logits,
            y0_pred,
            y1_pred
        )
import sys
sys.path.append("../")

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.blocks import EmbeddingAdder

'''
3-head architecture: treatment and potential outcomes y_0 y_1
'''
class MIMIC_Transformer(nn.Module):
    
    def __init__(
        self, 
        cfg,
        vocab_size: int,
        batch_first:bool = True,
        padding_idx: int = 0):
        
        super().__init__()
        
        self.diag_embed = nn.Embedding(num_embeddings=vocab_size,
                                       embedding_dim= cfg.MODEL.EMBEDDING_DIM,
                                       padding_idx=padding_idx)
        
        self.gender_embed = nn.Embedding(num_embeddings = 2, embedding_dim=1)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model = cfg.MODEL.EMBEDDING_DIM + 2, nhead= 2)
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = cfg.MODEL.TRANS_DEPTH,
        )
        
        self.embedding_adder = nn.Sequential(
            EmbeddingAdder(),
            nn.Dropout(cfg.MODEL.DROPOUT_P),
        )

        self.y0_fc = nn.Sequential(
            nn.Linear(cfg.MODEL.EMBEDDING_DIM + 2, cfg.MODEL.FC_HIDDEN_SIZE),
            nn.BatchNorm1d(cfg.MODEL.FC_HIDDEN_SIZE),
            nn.Dropout(cfg.MODEL.DROPOUT_P),
            nn.ReLU(),
            nn.Linear(cfg.MODEL.FC_HIDDEN_SIZE, 1)
        )

        self.y1_fc = nn.Sequential(
            nn.Linear(cfg.MODEL.EMBEDDING_DIM + 2, cfg.MODEL.FC_HIDDEN_SIZE),
            nn.BatchNorm1d(cfg.MODEL.FC_HIDDEN_SIZE),
            nn.Dropout(cfg.MODEL.DROPOUT_P),
            nn.ReLU(),
            nn.Linear(cfg.MODEL.FC_HIDDEN_SIZE, 1)
        )
        
        self.tr_fc = nn.Linear(cfg.MODEL.EMBEDDING_DIM + 2, 1) # treatment
        self.bce_loss_fn = nn.BCEWithLogitsLoss() #
        
    def forward(self, x_diag, x_age, x_gender, t, yf, y0, y1):

        batch_size, max_seq_length, _ = x_diag.shape
        
        # Embeddings
        x_embed = self.diag_embed(x_diag)

        x_gender = (
            self.gender_embed(x_gender)
            .unsqueeze(1)
            .expand(batch_size, max_seq_length, -1)
        )

        x_age =  x_age.unsqueeze(1).expand(batch_size, max_seq_length, -1)
        
        # Summing embedded values 
        x_embed_sum = self.embedding_adder(x_embed)
        x_embed_sum = torch.cat([x_embed_sum, x_gender, x_age],dim=2)
        
        x_trans = self.transformer(x_embed_sum)
        x_seq= x_trans.mean(1)
        
        y0_pred = self.y0_fc(x_seq).view(-1)
        y1_pred = self.y1_fc(x_seq).view(-1)
        yf_pred =  torch.where(t == 0, y0_pred, y1_pred)
        t_logits = self.tr_fc(x_seq).view(-1)
            
        t_loss = self.bce_loss_fn(t_logits, t)
        factual_loss = self.bce_loss_fn(yf_pred, yf)

        loss =  t_loss + factual_loss
        
        return (
            loss,
            yf_pred,
            y0_pred,
            y1_pred,
            t_logits
        )

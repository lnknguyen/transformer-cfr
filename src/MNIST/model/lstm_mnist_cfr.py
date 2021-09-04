import sys
sys.path.append("../")

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.blocks import EmbeddingAdder
from utils.loss_func import lindisc, wass

'''
3-head architecture: treatment and potential outcomes y_0 y_1
'''
class MNIST_LSTM(nn.Module):
    
    def __init__(
        self, 
        cfg,
        vocab_size: int,
        batch_first:bool=True):
        
        super().__init__()
        
        self.cfg = cfg
        self.embedding = nn.Embedding(vocab_size, cfg.MODEL.EMBEDDING_DIM)

        self.lstm = nn.LSTM(
            input_size= cfg.MODEL.EMBEDDING_DIM,
            hidden_size= cfg.MODEL.LSTM_HIDDEN_SIZE,
            batch_first= batch_first,
            dropout= cfg.MODEL.DROPOUT_P,
            num_layers= cfg.MODEL.LSTM_NUM_LAYER,
            bidirectional = True
        )
        
        self.embedding_adder = nn.Sequential(
            EmbeddingAdder(),
            nn.Dropout(cfg.MODEL.DROPOUT_P),
        )

        self.y0_fc = nn.Sequential(
            nn.Linear(cfg.MODEL.LSTM_HIDDEN_SIZE * 2, cfg.MODEL.FC_HIDDEN_SIZE),
            nn.BatchNorm1d(cfg.MODEL.FC_HIDDEN_SIZE),
            nn.Dropout(cfg.MODEL.DROPOUT_P),
            nn.ReLU(),
            nn.Linear(cfg.MODEL.FC_HIDDEN_SIZE, 1)
        )

        self.y1_fc = nn.Sequential(
            nn.Linear(cfg.MODEL.LSTM_HIDDEN_SIZE * 2, cfg.MODEL.FC_HIDDEN_SIZE),
            nn.BatchNorm1d(cfg.MODEL.FC_HIDDEN_SIZE),
            nn.Dropout(cfg.MODEL.DROPOUT_P),
            nn.ReLU(),
            nn.Linear(cfg.MODEL.FC_HIDDEN_SIZE, 1)
        )
        
        self.tr_fc = nn.Linear(cfg.MODEL.LSTM_HIDDEN_SIZE * 2, 1) # treatment
        self.bce_loss_fn = nn.BCEWithLogitsLoss() #
        self.mse_loss = nn.MSELoss()
        
    def forward(self, xs_1,xs_2, xs_3, t, yf):

        batch_size, max_seq_length = xs_1.shape
        
        #Input: [batch_size, seq_length]
        #Output: [batch_size, seq_length, embedding_dimension]
        x1_embed = self.embedding(xs_1)
        x2_embed = self.embedding(xs_2)
        x3_embed = self.embedding(xs_3)

        x_embed = self.embedding_adder(torch.stack([x1_embed, x2_embed, x3_embed], dim=2))
       
        # x shape needs go be [batch_size, seq_len, embedding_dimension] 
        # before fetching into lstm layer
        x_out, _ = self.lstm(x_embed)
        x_out = x_out.contiguous().view(batch_size, max_seq_length, -1)

        # get last relevant hidden unit
        x_seq = x_out[torch.arange(batch_size), -1]
    
        # x_seq shape: [batch_size, batch_size]
        y0_pred = self.y0_fc(x_seq).view(-1)
        y1_pred = self.y1_fc(x_seq).view(-1)
        t_logits = self.tr_fc(x_seq).view(-1)

        yf_pred =  torch.where(t == 0, y0_pred, y1_pred)
        t_loss = self.bce_loss_fn(t_logits, t)
        
        if self.cfg.SIM.OUTPUT_TYPE == 'binary':
            y_loss = self.bce_loss_fn(yf_pred, yf)
        else:
            y_loss = self.mse_loss(yf_pred, yf)
            
        #t_loss = lindisc(x_seq.cpu().detach().numpy(),0.5, t.cpu().detach().numpy())
        reg_lambda = 1e-2
        loss = t_loss + reg_lambda* y_loss
        
        return (
            loss,
            yf_pred,
            y0_pred,
            y1_pred,
            t_logits
        )

import torch
import torch.nn as nn
import torch.nn.functional as F


class Full_Attention(nn.Module):
    def __init__(self, mask_flag=False, output_attention=True, attention_dropout=0.1):
        super(Full_Attention, self).__init__()
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, Q, K, V):
        B, L, H, E = Q.shape #batch, seq_len, head, embed
        _, S, _, D = V.shape
        scale = 1. / np.sqrt(E)

        Q = Q.permute(0, 2, 3, 1)  # size = [B, H, E, L]
        K = K.permute(0, 2, 3, 1)
        V = V.permute(0, 2, 3, 1)

        scores = torch.einsum('bhex,bhey -> bhxy', Q, K) * scale

        atten= F.softmax(scores.abs(), dim=-1)
        #atten = self.dropout(atten)
        if self.mask_flag:
            mask = torch.triu(torch.ones(S, S), 1).bool().to(atten.device)
            atten = atten.masked_fill(mask, 0)
            
        atten = torch.einsum("bhxy,bhey->bhex", atten, V)
        
        out = atten.permute(0, 3, 1, 2)       
        
        if self.output_attention == False:
            return out
        else:
            return out, atten
            
class Attention_Layer(nn.Module):
    def __init__(self, attention, enc_dim, heads, dropout):
        super(Attention_Layer, self).__init__()
        self.heads = heads
        self.enc_dim = enc_dim
        self.head_dim = enc_dim // heads
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** 0.5
        self.heads=heads

        self.Q_linear = nn.Linear(self.enc_dim, self.head_dim*self.heads)
        self.K_linear = nn.Linear(self.enc_dim, self.head_dim*self.heads)
        self.V_linear = nn.Linear(self.enc_dim, self.head_dim*self.heads)

        self.attention=attention

        self.fc_out = nn.Linear(self.head_dim*self.heads, enc_dim)

    def forward(self, Q, K, V):
        B = Q.shape[0] #batch size
        Q_len, K_len, V_len = Q.shape[1], K.shape[1], V.shape[1] #seq_len

        Q=self.Q_linear(Q).view(B, Q_len, self.heads, self.head_dim)
        K=self.K_linear(K).view(B, K_len, self.heads, self.head_dim)
        V=self.V_linear(V).view(B, V_len, self.heads, self.head_dim)

        out,atten = self.attention(Q, K, V)

        return self.fc_out(out),atten
            
            

class FeedForward(nn.Module):
    def __init__(self, enc_dim, hidden_conv_dim, dropout = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(enc_dim, hidden_conv_dim,kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_conv_dim, enc_dim,kernel_size=1),
            nn.Dropout(dropout)
        )
    def forward(self, x): 
        x = self.net(x)
        return x
        
        

class Encoder_layer(nn.Module):
    def __init__(self, attention,feedforward ,enc_dim, dropout, output_attention=False):
        super(Encoder_layer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        self.norm1 = nn.LayerNorm(enc_dim)
        self.norm2 = nn.LayerNorm(enc_dim)
        self.attention = attention
        self.feedforward = feedforward
        self.output_attention=output_attention

    def forward(self, x):
        if self.output_attention:
            new_x, atten = self.attention(x, x, x)
            new_x = self.norm1(x + self.dropout(new_x))
            new_x = self.norm2(new_x + self.feedforward(new_x))
            return new_x, atten
        else:
            new_x = self.attention(x, x, x)
            new_x = self.norm1(x + self.dropout(new_x))
            new_x = self.norm2(new_x + self.feedforward(new_x))
            return new_x
            

class Transformer(nn.Module):
    def __init__(self,enc_dim=512,heads=8,hidden_ff_dim=2048,e_layers,dropout=0.2):
        super(Model, self).__init__()
        
        self.attention=Attention_Layer(Full_Attention(attention_dropout=dropout),enc_dim,heads,dropout)
        self.feedforward=FeedForward(enc_dim, hidden_ff_dim, dropout)
        #self.vocab_size=4**args.kmer
        #self.embedding = nn.Embedding(self.vocab_size, args.enc_dim)    
        self.encoder = nn.ModuleList([Encoder_layer(self.attention,self.feedforward,enc_dim,dropout) for _ in range(e_layers)])
        self.fc1_out = nn.Linear(enc_dim,1)

    def forward(self, x):
        #x = self.embedding(x)
        for layer in self.encoder:
            x = layer(x)
        x = self.fc1_out(x)
        return x.squeeze(-1)

import torch
import torch.nn as nn

from ..utils.util import normQs, Qs2Rs


class CnnLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=5, stride=1, padding='same'):
        super(CnnLayer, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.batch_norm = nn.BatchNorm1d(out_channel)
        self.activation = nn.GELU()
    
    def forward(self, encodings):
        encodings = self.conv(encodings)
        encodings = self.batch_norm(encodings)
        encodings = self.activation(encodings)
        return encodings


class Decoder(nn.Module):
    def __init__(self, resnet_config):
        super(Decoder, self).__init__()
        self.resnet_config = resnet_config
        self.layer_norm = nn.LayerNorm(self.resnet_config.hidden_dim)
        self.hidden_channels = [self.resnet_config.hidden_size] + self.resnet_config.channel_multipliers + [self.resnet_config.hidden_size]
        self.layers = nn.ModuleList()

        for index in range(len(self.hidden_channels)-1):
            self.layers.extend([
                CnnLayer(self.hidden_channels[index], self.hidden_channels[index+1])
            ])
        self.head = nn.Linear(self.resnet_config.hidden_size, self.resnet_config.output_dim)
        self.sc_predictor = nn.Linear(self.resnet_config.hidden_size, 1)

    def forward(self, encodings):
        B, L = encodings.shape[0], encodings.shape[1]
        x = encodings
        for layer in self.layers:
            encodings = torch.transpose(encodings, 2, 1)
            #encodings = [B, h, W, H]
            encodings = layer(encodings)
            encodings = torch.transpose(encodings, 2, 1)
        
        encodings = x + encodings
        logits = self.head(encodings)
        logits = logits.reshape(B, L, 2, 3)

        Ts = logits[:,:,0,:] * 10.0
        Qs = logits[:,:,1,:]
        Qs = torch.cat((torch.ones((B, L, 1),device=Qs.device), Qs),dim=-1)
        Qs = normQs(Qs)
        Rs = Qs2Rs(Qs)

        Tout = Ts[..., None]
        #alpha = self.sc_predictor(encodings)

        return Rs, Tout
    
    
class AttenionBasedDecoder(nn.Module):
    def __init__(self, config):
        super(AttenionBasedDecoder, self).__init__()

        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.extend([self.build_attention_block(config.head_dim, config.num_heads, config.p_drop)])

        self.rota_trans_head = nn.Linear(config.hidden_dim, config.output_dim)
        #self.sc_predictor = nn.Linear(self.resnet_config.hidden_size, 1) #predict alpha torsion angles
        
    def build_attention_block(self, embed_dim, num_heads, p_drop, add_bias_kv=False, add_zero_attn=False):
        return nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=p_drop,
            bias=True,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

    def forward(self, hidden_state):
        B, L = hidden_state.shape[:2]
        
        res = hidden_state
        for layer in self.layers:
            hidden_state, _ = layer(hidden_state, hidden_state, hidden_state) #q,k,v
        
        hidden_state = res + hidden_state
        logits = self.rota_trans_head(hidden_state)
        logits = logits.reshape(B, L, 2, 3)

        Ts = logits[:,:,0,:] * 10.0
        Qs = logits[:,:,1,:] #quartenions
        Qs = torch.cat((torch.ones((B, L, 1),device=Qs.device), Qs), dim=-1)
        Qs = normQs(Qs)
        Rs = Qs2Rs(Qs)

        #Ts = Ts[..., None]
        #alpha = self.sc_predictor(encodings)

        return Rs, Ts
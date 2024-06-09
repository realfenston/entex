import esm
import torch
import torch.nn as nn
import torch.nn.functional as F
from esm.inverse_folding.util import CoordBatchConverter
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from .auxiliary_modules import FeedForwardNetwork
from ..utils.util import entropy_loss_fn


class Encoder(nn.Module):
    def __init__(self, encode_dim, hidden_dim, rff=4, p_drop=0.1, freeze_model=False):
        super(Encoder, self).__init__()
        self.model, self.alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False
        
        #manually mock a trainable encoder, while freezing params in the original ESM encoder for memory issue
        self.fc = nn.Linear(encode_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p_drop)
        self.ffn = FeedForwardNetwork(hidden_dim, rff, p_drop)
        
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, coords_list):
        batch_converter = CoordBatchConverter(self.alphabet, truncation_seq_length=1024)
        batch_coords, confidence, _, _, padding_mask = (
            batch_converter([(coord, None, None) for coord in coords_list], device=coords_list.device)
        )
        encoder_out = self.model.encoder(batch_coords, padding_mask, confidence)
        
        feat = encoder_out['encoder_out'][0].permute(1, 0, 2)[:,1:-1] #this is due to esm legacy issue
        x = torch.tensor(feat)
        x = self.fc(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.norm(x)
        x = self.norm(self.ffn(x))
        return x


class CNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=5, stride=1, padding='same'):
        super(CNNLayer, self).__init__()
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
                CNNLayer(self.hidden_channels[index], self.hidden_channels[index+1])
            ])
        self.head = nn.Linear(self.resnet_config.hidden_size, self.resnet_config.output_dim)
        self.sc_predictor = nn.Linear(self.resnet_config.hidden_size, 1)
        # self.norm_fn = get_norm_layer(train=train, dtype=self.dtype, norm_type=self.norm_type)

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

        T = logits[:,:,0,:] / 10
        R = logits[:,:,1,:] / 100.0

        Qnorm = torch.sqrt(1 + torch.sum(R*R, dim=-1))
        qA, qB, qC, qD = 1/Qnorm, R[:,:,0]/Qnorm, R[:,:,1]/Qnorm, R[:,:,2]/Qnorm

        Rout = torch.zeros((B, L, 3, 3)).to(R.device)
        Rout[:,:,0,0] = qA*qA+qB*qB-qC*qC-qD*qD
        Rout[:,:,0,1] = 2*qB*qC - 2*qA*qD
        Rout[:,:,0,2] = 2*qB*qD + 2*qA*qC
        Rout[:,:,1,0] = 2*qB*qC + 2*qA*qD
        Rout[:,:,1,1] = qA*qA-qB*qB+qC*qC-qD*qD
        Rout[:,:,1,2] = 2*qC*qD - 2*qA*qB
        Rout[:,:,2,0] = 2*qB*qD - 2*qA*qC
        Rout[:,:,2,1] = 2*qC*qD + 2*qA*qB
        Rout[:,:,2,2] = qA*qA-qB*qB-qC*qC+qD*qD

        Tout = T.unsqueeze(2)
        alpha = self.sc_predictor(encodings)

        return Rout, Tout, alpha
    

class LanguageQuantizer(nn.Module):
    def __init__(self, quantizer_config, codebook, hidden_dim=256):
        super(LanguageQuantizer, self).__init__()
        self.quantizer_config = quantizer_config
        self.codebook = codebook
        self.input_to_latent = nn.Linear(hidden_dim, quantizer_config.quantizer_hidden_dim)
        self.code_to_latent = nn.Linear(self.quantizer_config.quantizer_latent_dim, quantizer_config.quantizer_hidden_dim)
        
    @staticmethod
    def normalize_func(x, axis=None, eps=1e-6, use_l2_normalize=True):
        if use_l2_normalize:
            return x * torch.rsqrt((x * x).sum(dim=axis, keepdim=True) + eps)
        else:
            return x
        
    @staticmethod
    def squared_euclidean_distance(a, b, b2=None, dot_product=False):
        if dot_product:
            return torch.matmul(a, b.T)
        if b2 is None:
            b2 = torch.sum(b.T**2, axis=0, keepdims=True)
        a2 = torch.sum(a**2, axis=1, keepdims=True)
        ab = torch.matmul(a, b.T)
        d = a2 - 2 * ab + b2
        return d

    def forward(self, x, train=True):
        l2_normalize = lambda x, axis=1: LanguageQuantizer.normalize_func(
            x, axis=axis, use_l2_normalize=self.quantizer_config.l2_normalize
        )

        codebook = self.codebook.detach()
        codebook_size = codebook.shape[0]
        if self.quantizer_config.strawman_codebook:
            strawman_codebook = self.param(
                "strawman_codebook",
                torch.nn.init.normal_(torch.empty((codebook_size, self.quantizer_config.quantizer_latent_dim)), mean=0.02),
            )
            # strawman_codebook = torch.tensor(strawman_codebook, dtype=self.dtype)
            latent_input = self.input_to_latent(x.reshape((-1, x.shape[-1])))
            latent_input = l2_normalize(latent_input, axis=1)
            sg_strawman_codebook = (
                l2_normalize(strawman_codebook, axis=1).detach()
            )
            distances = torch.reshape(
                LanguageQuantizer.squared_euclidean_distance(latent_input, sg_strawman_codebook,
                dot_product=self.quantizer_config.dot_product),
                x.shape[:-1] + (codebook_size,),
            )
        else:
            latent_input = self.input_to_latent(torch.reshape(x, (-1, x.shape[-1])))
            
            #TODO axis=1 or axi=-1?
            latent_input = l2_normalize(latent_input, axis=-1)
            latent_codebook = self.code_to_latent(codebook)
            latent_codebook = l2_normalize(latent_codebook, axis=1)
            sg_latent_codebook = latent_codebook.detach()

            distances = torch.reshape(LanguageQuantizer.squared_euclidean_distance(latent_input, sg_latent_codebook, dot_product=self.quantizer_config.dot_product), x.shape[:-1] + (codebook_size,),)

        encoding_indices = torch.topk(distances, k=self.quantizer_config.top_k_value, dim=-1, largest=False)[1]

        encoding_indices, encodings, quantized = self.get_encoding_quantized(
            encoding_indices, codebook_size
        )

        """codebook_usage = torch.sum(encodings, axis=(0, 1)) > 0
        codebook_usage = torch.sum(codebook_usage).to(torch.float32) / codebook_size
        if self.quantizer_config.top_k_avg:
            codebook_usage = codebook_usage / self.config.top_k_value"""
        result_dict = dict()
        if train:
            #used in original paper, but quantized and x have a shape mismatch
            #please ensure hidden dim == codebook dim here
            result_dict = self.get_train_loss(quantized, x, distances) 

            if self.quantizer_config.strawman_codebook:
                strawman_quantized = self.quantize_strawman(encodings)
                strawman_result_dict = self.get_train_loss(
                    strawman_quantized, self.input_to_latent(x), distances
                )
                for k, v in result_dict.items():
                    result_dict[k] = v + strawman_result_dict[k]
            else:
                latent_quantized = self.code_to_latent(quantized) #linear layer
                latent_result_dict = self.get_train_loss(
                    latent_quantized, self.input_to_latent(x), distances
                )
                for k, v in result_dict.items():
                    result_dict[k] = v + latent_result_dict[k]

            quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings.reshape(-1, encodings.shape[-1]), axis=0)
        log_perplexity = -torch.sum(avg_probs * torch.log(avg_probs + 1e-6))
        perplexity = torch.exp(log_perplexity)

        if "quantizer_loss" in result_dict:
            result_dict["quantizer_loss"] = (
                result_dict["quantizer_loss"]
                + self.quantizer_config.quantizer_loss_perplexity * log_perplexity
            )
        result_dict.update(
            {
                "encodings": encodings,
                "encoding_indices": encoding_indices,
                "raw": x,
                "perplexity": perplexity
            }
        )
        return quantized, result_dict

    def quantize(self, z):
        codebook = self.codebook.detach()
        return torch.matmul(z, codebook)

    def get_codebook(self):
        return self.codebook

    def decode_ids(self, ids):
        return torch.index_select(self.codebook, 0, ids,)

    def quantize_strawman(self, z):
        return torch.matmul(z, self.variables["params"]["strawman_codebook"])

    def get_train_loss(self, quantized, x, distances):
        e_latent_loss = (
            torch.mean((((quantized.detach()) - x) ** 2))
            * self.quantizer_config.quantizer_loss_commitment
        ).to(torch.float32)
        q_latent_loss = torch.mean((quantized - x.detach()) ** 2)
        entropy_loss = torch.tensor(0.0).to(q_latent_loss.device)
        if self.quantizer_config.quantizer_loss_entropy != 0:
            entropy_loss = (
                entropy_loss_fn(
                    -distances,
                    loss_type=self.quantizer_config.entropy_loss_type,
                    temperature=self.quantizer_config.entropy_temperature,
                )
                * self.quantizer_config.quantizer_loss_entropy
            )
        #e_latent_loss = torch.tensor(e_latent_loss).to(e_latent_loss.dev·ice).to(torch.float32)
        loss = e_latent_loss + q_latent_loss + entropy_loss

        result_dict = dict(
            quantizer_loss=loss,
            e_latent_loss=e_latent_loss,
            q_latent_loss=q_latent_loss,
            entropy_loss=entropy_loss,
        )
        return result_dict

    def get_encoding_quantized(self, encoding_indices, codebook_size, train=True):
        if self.quantizer_config.top_k_rnd:
            if train:
                encoding_indices = torch.random.choice(encoding_indices, axis=-1)
            else:
                encoding_indices = encoding_indices[..., 0]
            encodings = F.one_hot(
                encoding_indices, codebook_size
            )
            quantized = self.quantize(encodings)
        elif self.quantizer_config.top_k_avg:
            encodings = F.one_hot(encoding_indices, codebook_size)
            quantized = self.quantize(encodings)
            quantized = torch.mean(quantized, axis=-2)
            encoding_indices = encoding_indices[..., 0]
        else:
            encoding_indices = encoding_indices[..., 0]
            encodings = F.one_hot(encoding_indices, codebook_size).to(torch.float32)
            quantized = self.quantize(encodings)
        return encoding_indices, encodings, quantized
    

class Add_Pos(nn.Module):
    def __init__(self, config):
        super(Add_Pos, self).__init__()
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_dim,
        )
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()
    
    def init_weights(self):
        self.position_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        #self.token_type_embeddings.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, inputs_embeds, token_type_ids, position_ids):
        position_embeds = self.position_embeddings(position_ids.long())
        #token_type_embeddings = self.token_type_embeddings(token_type_ids.long())
        hidden_states = inputs_embeds + position_embeds

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
    

class LanguageModel(nn.Module):
    def __init__(self, pretrained_config, config, pretrained_codebook, pretrained_tokenizer):
        super(LanguageModel, self).__init__()
        self.embeddings = Add_Pos(config)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_path)
        self.tokenizer = pretrained_tokenizer
        self.word_embeddings = pretrained_codebook
        self.pretrained_config = pretrained_config

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(
        self,
        hidden_states,
        input_ids,
        attention_mask,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # make sure `token_type_ids` is correctly initialized when not passed
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids).to(hidden_states.device)

        # make sure `position_ids` is correctly initialized when not passed
        if position_ids is None:
            position_ids = LanguageModel.create_position_ids_from_input_ids(
                input_ids, self.pretrained_config.pad_token_id
            )

        hidden_states = self.embeddings(hidden_states, token_type_ids, position_ids)
        attention_mask = attention_mask.to(hidden_states.device)

        """outputs = self.model(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )"""
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)

        hidden_states = outputs[0]
        if self.pretrained_config.tie_word_embeddings:
            shared_embedding = self.word_embeddings.to(hidden_states.device)
        else:
            shared_embedding = None

        #Compute the prediction scores
        #logits = self.lm_head(hidden_states, shared_embedding=shared_embedding)
        logits = torch.matmul(hidden_states, shared_embedding.T)

        if not return_dict:
            return logits, outputs[1:]

        return {
            "logits": logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
            "last_hidden_states": hidden_states,
        }
    
    @staticmethod
    def create_position_ids_from_input_ids(input_ids, pad_token_id):
        mask = input_ids.ne(pad_token_id)
        position_ids = torch.cumsum(mask, dim=1) * mask
        return position_ids.long()
    

class XYZConverter(nn.Module):
    def __init__(self):
        super(XYZConverter, self).__init__()
        self.basexyzs = torch.tensor([(-0.5272, 1.3593, 0.000, 1),
                                      (0.000, 0.000, 0.000, 1),
                                      (1.5233, 0.000, 0.000, 1)])

    def compute_all_atom(self, Rs, Ts):
        B, L = Rs.shape[:2]
        RTF0 = torch.eye(4).repeat(B, L, 1, 1).to(device=Rs.device)
        RTF0[:,:,:3,:3] = Rs
        RTF0[:,:,:3,3] = Ts.squeeze(2)
        #RTframes = torch.stack((RTF0), dim=2)
        basexyzs = self.basexyzs[None, None, ...].repeat(B, L, 1, 1).to(Rs.device)
        xyzs = torch.einsum('brij,brmj->brmi', RTF0, basexyzs)
        return RTF0, xyzs[...,:3]
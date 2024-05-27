import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .loss import calc_str_loss
from .lr_schedulers import get
from .module import Decoder, Encoder, LanguageModel, LanguageQuantizer, XYZConverter


class LQAE_model(nn.Module):
    def __init__(self, model_config):
        super(LQAE_model, self).__init__()
        self.model_config = model_config
        self.lang_model, self.codebook, self.mask_code, self.tokenizer = self.config_language_model()
        self.codebook.requires_grad = False

        self.encoder = Encoder(self.model_config.bbencoder.encode_dim, self.model_config.hidden_dim)
        self.quantizer = LanguageQuantizer(quantizer_config=self.model_config.quantizer, codebook=self.codebook, hidden_dim=self.model_config.hidden_dim)
        self.decoder = Decoder(resnet_config=self.model_config.resnet)

    def config_language_model(self):
        self.bert_config = self.model_config.bert
        if self.bert_config.bert_name == 'allenai/scibert_scivocab_uncased':
            pretrained_bert = AutoModel.from_pretrained(self.bert_config.bert_path)
        else:
            raise ValueError("Unmatched language model version, please choose among (scibert,)")
        language_model = LanguageModel(pretrained_bert.config, self.bert_config)
        codebook = pretrained_bert.embeddings.word_embeddings.weight

        tokenizer = AutoTokenizer.from_pretrained(self.bert_config.bert_path)
        mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        mask_code = codebook[mask_token_id].detach()
        return language_model, codebook, mask_code, tokenizer
    
    @staticmethod
    def random_mask(x, ratio):
        random_values = torch.empty(x.shape[:2]).uniform_(0, 1).to(x.device)
        mask = random_values < ratio
        return mask.to(torch.bool)

    @staticmethod
    def random_ratio_mask(x, min_ratio, max_ratio):
        ratio = torch.empty(x.shape[:2]).uniform_(min_ratio, max_ratio).to(x.device)
        return LQAE_model.random_mask(x, ratio)
    
    def languge_model_encode_decode(
        self,
        input_code,
        input_ids,
        ratio={},
        output_hidden_states=False,
    ):
        input_shape = input_code.shape
        if len(input_code.shape) == 4:
            input_code = torch.reshape(
                input_code, (input_code.shape[0], -1, input_code.shape[-1])
            )
            input_ids = torch.reshape(input_ids, (input_ids.shape[0], -1))

        min_ratio = ratio.get("min_ratio", self.bert_config.bert_min_ratio)
        max_ratio = ratio.get("max_ratio", self.bert_config.bert_max_ratio)
        assert min_ratio <= max_ratio, "min_ratio must be less than max_ratio"
        
        use_mask = LQAE_model.random_ratio_mask(
            torch.zeros((input_code.shape[0], input_code.shape[1])).to(input_code.device),
            min_ratio,
            max_ratio,
        ).to(input_code.device).to(torch.bool)

        mask_code = self.mask_code.to(input_code.device)
        input_code = torch.where(
            use_mask[..., None], mask_code[None, None, ...], input_code
        )

        attention_mask = torch.ones((input_code.shape[0], input_code.shape[1])).to(input_code.device)
        bert_output = self.lang_model(
            input_code,
            input_ids,
            attention_mask,
            output_hidden_states=output_hidden_states,
        )

        if self.model_config.bert.use_bert_ste:
            logits = bert_output['logits']
            decoding_indices = torch.argmax(logits, axis=-1).to(logits.device)
            codebook_size = self.codebook.shape[0]
            encodings = F.one_hot(decoding_indices, codebook_size)
            argmax_code = torch.matmul(encodings.to(torch.float32), self.codebook)
            softmax_code = torch.matmul(F.softmax(logits, dim=-1), self.codebook)
            output = softmax_code + (argmax_code - softmax_code).detach()
            output = output.reshape(input_shape)
        else:
            output = bert_output['last_hidden_states']
            output = output.reshape(input_shape)

        logits = bert_output['logits']
        
        B, L = logits.shape[:2]
        bert_loss = F.cross_entropy(
            logits.reshape(B*L,logits.shape[-1]), 
            F.one_hot(input_ids, logits.shape[-1]).float().reshape(B*L,logits.shape[-1]),
            reduction='none'
        ).reshape(B, L)
        
        if self.model_config.bert.bert_loss_mask_only:
            bert_loss = bert_loss * use_mask
            bert_loss = torch.sum(bert_loss, axis=1) / torch.sum(use_mask, axis=1)

        bert_loss = torch.mean(bert_loss) * self.model_config.bert.bert_mask_loss_weight
        language_model_output = {
            "bert_logits": bert_output['logits'],
            "bert_hidden_states": bert_output['hidden_states'],
            "bert_loss": bert_loss,
        }
        return output, language_model_output
    
    def encode(self, structure):
        encoded_feature = self.encoder(structure)
        quantized, result_dict = self.quantizer(encoded_feature)
        return quantized, result_dict
    
    def decode(self, x):
        reconstructed = self.decoder(x)
        return reconstructed
    
    def forward(self, structure, ratio={}):
        quantized, result_dict = self.encode(structure)
        bert_quantized, language_model_output = self.languge_model_encode_decode(
            quantized, result_dict["encoding_indices"], ratio
        )
        result_dict = {**result_dict, **language_model_output}
        structure_output = self.decoder(quantized)
        bert_channel_structure_output = self.decoder(bert_quantized)
        output = {
            "structure_output": structure_output,
            "bert_channel_structure_output": bert_channel_structure_output,
        }
        return output, result_dict
    

class LQAE(pl.LightningModule):
    def __init__(self, config=None):
        super(LQAE, self).__init__()
        self.model_config = config.model
        self.optimizer_config = config.optimizer
        self.lqae_model = LQAE_model(self.model_config)
        self.xyz_converter = XYZConverter()

    def get_loss(self, result_dict, reconX, nativeX, mask, train=True):
        if "bert_loss" in result_dict:
            bert_loss = result_dict["bert_loss"]
        else:
            bert_loss = 0.0
        if train:
            quantizer_loss = result_dict["quantizer_loss"]
        else:
            quantizer_loss = 0.0
        
        recon_loss, _, _ = calc_str_loss(reconX, nativeX, mask)
        return quantizer_loss, bert_loss, recon_loss
        
    def forward(self, coords):
        return self.lqae_model(coords)

    def training_step(self, batch, batch_idx):
        B = batch['X'].shape[0]
        X, mask = batch['X'], batch['mask']
        output, result_dict = self(X)

        Rs, Ts, alphas = output['structure_output'] #alphas, or the torsion angles, are not required for backbone generation
        _, xyz = self.xyz_converter.compute_all_atom(Rs, Ts)
        #fape_loss = self.compute_general_FAPE(xyz, X, mask)
        #quantizer_loss, e_latent_loss, q_latent_loss, entropy_loss = result_dict['quantizer_loss'], result_dict['e_latent_loss'], result_dict['q_latent_loss'], result_dict['entropy_loss']

        #total_loss = fape_loss + quantizer_loss + e_latent_loss + q_latent_loss + entropy_loss
        quantizer_loss, bert_loss, recon_loss = self.get_loss(result_dict, xyz, X, mask)
        total_loss = quantizer_loss + bert_loss + recon_loss
        #total_loss = quantizer_loss
        #total_loss = bert_loss
        #total_loss = recon_loss
        
        total_loss = total_loss.mean()
        self.log("train_loss", total_loss, sync_dist=True, batch_size=B)
        #self.log("train_perplexity", perplexity, prog_bar=True, sync_dist=True)

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        B = batch['X'].shape[0]
        with torch.no_grad():
            X, mask = batch['X'], batch['mask']
            output, result_dict = self(X)
            Rs, Ts, alphas = output['structure_output'] #alphas, or the torsion angles, are not required for backbone generation
            _, xyz = self.xyz_converter.compute_all_atom(Rs, Ts)
            quantizer_loss, bert_loss, recon_loss = self.get_loss(result_dict, xyz, X, mask)
            total_loss = quantizer_loss + bert_loss + recon_loss
            total_loss = total_loss.mean()
            perplexity = total_loss.float().exp().mean()
            self.log("validation_loss", total_loss, sync_dist=True, batch_size=B)
            self.log("validation_perplexity", perplexity, sync_dist=True, batch_size=B)
            return total_loss

    def configure_optimizers(self):
        no_decay = ["norm", "LayerNorm"]
        
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if any(nd in name for nd in no_decay):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": self.optimizer_config.weight_decay,
            },
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        if self.optimizer_config.name == "adam":
            optimizer_type = torch.optim.AdamW
        else:
            raise ValueError("Un-recognized optimizer type, please choose among (adam,)")
        optimizer = optimizer_type(
            optimizer_grouped_parameters,
            lr=self.optimizer_config.learning_rate,
            betas=self.optimizer_config.adam_betas,
        )
        scheduler = get(self.optimizer_config.lr_scheduler)(
            optimizer,
            self.optimizer_config.warmup_steps,
            self.optimizer_config.max_steps,
        )
        scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_dict]
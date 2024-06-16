import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from ..modules.base_modules import XYZConverter
from ..lr_schedulers import get as lr_get
from ..modules.module_opt import Encoder, LanguageModel, LanguageQuantizer
from ..modules.decoder_module import AttenionBasedDecoder
from ..utils.loss import calc_str_loss


class LQAE_model(nn.Module):
    def __init__(self, model_config):
        super(LQAE_model, self).__init__()
        self.model_config = model_config
        self.lang_model, self.codebook, self.tokenizer = self.config_language_model()
        self.encoder = Encoder(self.model_config.backbone.encode_dim, self.model_config.hidden_dim)
        self.quantizer = LanguageQuantizer(quantizer_config=self.model_config.quantizer, codebook=self.codebook, hidden_dim=self.model_config.hidden_dim)
        self.decoder = AttenionBasedDecoder(config=self.model_config.decoder)

    def config_language_model(self):
        #self.opt_config = self.model_config.opt
        opt_config = self.model_config.opt
        if opt_config.name == 'opt-125M':
            pretrained_model = AutoModelForCausalLM.from_pretrained(opt_config.model_path)
        else:
            raise ValueError("Unmatched language model version, please choose among (opt-125M, llama2)")
        codebook = pretrained_model.get_input_embeddings().weight #fix the pretrained model codebook
        tokenizer = AutoTokenizer.from_pretrained(opt_config.model_path)
        
        language_model = LanguageModel(pretrained_model.config, opt_config, codebook, tokenizer)
        return language_model, codebook, tokenizer
    
    def languge_model_encode_decode(
        self,
        input_code,
        input_ids,
        output_hidden_states=False,
    ):
        input_shape = input_code.shape
        if len(input_code.shape) == 4:
            input_code = torch.reshape(
                input_code, (input_code.shape[0], -1, input_code.shape[-1])
            )
            input_ids = torch.reshape(input_ids, (input_ids.shape[0], -1))

        attention_mask = torch.ones((input_code.shape[0], input_code.shape[1])).to(input_code.device)
        lm_output = self.lang_model(
            input_code,
            input_ids,
            attention_mask,
            output_hidden_states=output_hidden_states,
        )

        if self.model_config.opt.use_opt_ste:
            logits = lm_output['logits']
            decoding_indices = torch.argmax(logits, axis=-1).to(logits.device)
            codebook = self.codebook.detach()
            codebook_size = codebook.shape[0]
            encodings = F.one_hot(decoding_indices, codebook_size)
            argmax_code = torch.matmul(encodings.to(torch.float32), codebook)
            softmax_code = torch.matmul(F.softmax(logits, dim=-1), codebook)
            output = softmax_code + (argmax_code - softmax_code).detach()
            output = output.reshape(input_shape)
        else:
            output = lm_output['last_hidden_states']
            output = output.reshape(input_shape)
        
        lm_loss = lm_output['loss'] * self.model_config.opt.ar_loss_weight
        
        language_model_output = {
            "lm_logits": lm_output['logits'],
            "lm_hidden_states": lm_output['hidden_states'],
            "lm_loss": lm_loss,
        }
        return output, language_model_output
    
    def encode(self, structure):
        encoded_feature = self.encoder(structure)
        quantized, result_dict = self.quantizer(encoded_feature)
        return quantized, result_dict
    
    def decode(self, x):
        reconstructed = self.decoder(x)
        return reconstructed
    
    def forward(self, structure):
        quantized, result_dict = self.encode(structure)
        
        #prepare sequence indices for opt loss
        lm_quantized, language_model_output = self.languge_model_encode_decode(
            quantized, result_dict["encoding_indices"]
        )
        result_dict = {**result_dict, **language_model_output}
        structure_output = self.decoder(quantized)
        lm_channel_structure_output = self.decoder(lm_quantized)
        output = {
            "structure_output": structure_output,
            "lm_channel_structure_output": lm_channel_structure_output,
        }
        return output, result_dict
    

class LQAE(pl.LightningModule):
    def __init__(self, model_config, optimizer_config):
        super(LQAE, self).__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.lqae_model = LQAE_model(self.model_config)
        self.xyz_converter = XYZConverter()

    def get_loss(self, result_dict, reconX, nativeX, mask, train=True):
        if "lm_loss" in result_dict:
            bert_loss = result_dict["lm_loss"]
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
        
        #TODO
        #X = X[:, :1, ...]
        output, result_dict = self(X)

        Rs, Ts = output['structure_output'] #alphas, or the torsion angles, are not required for backbone generation
        _, xyz = self.xyz_converter.compute_all_atom(Rs, Ts)
        #fape_loss = self.compute_general_FAPE(xyz, X, mask)
        #quantizer_loss, e_latent_loss, q_latent_loss, entropy_loss = result_dict['quantizer_loss'], result_dict['e_latent_loss'], result_dict['q_latent_loss'], result_dict['entropy_loss']

        quantizer_loss, bert_loss, recon_loss = self.get_loss(result_dict, xyz, X, mask)
        total_loss = quantizer_loss + bert_loss + recon_loss

        total_loss = total_loss.mean()
        self.log("quantizer_loss", quantizer_loss, sync_dist=True, batch_size=B)
        self.log("bert_loss", bert_loss, sync_dist=True, batch_size=B)
        self.log("recon_loss", recon_loss, sync_dist=True, batch_size=B)
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
            self.log("valid_loss", total_loss, sync_dist=True, batch_size=B)
            self.log("valid_perplexity", perplexity, sync_dist=True, batch_size=B)
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
            {"params": decay_params, "weight_decay": self.optimizer_config.weight_decay},
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
        scheduler = lr_get(self.optimizer_config.lr_scheduler)(
            optimizer,
            self.optimizer_config.warmup_steps,
            self.optimizer_config.max_steps,
        )
        scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_dict]
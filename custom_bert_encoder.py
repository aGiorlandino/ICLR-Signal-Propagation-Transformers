import copy
from torch import nn
from transformers.models.bert.modeling_bert import BertEncoder, BertConfig
from custom_bert_layer import CustomBertLayer  

class CustomBertEncoder(BertEncoder):
    def __init__(self, config: BertConfig, alpha_attention: float = 1.0, alpha_ffn: float = 1.0):
        super().__init__(config) 
        self.layer = nn.ModuleList([
            CustomBertLayer(config, alpha_attention=alpha_attention, alpha_ffn=alpha_ffn)
            for _ in range(config.num_hidden_layers)
        ])

        self.gradient_checkpointing = False


from transformers.models.bert.modeling_bert import BertModel, BertForMaskedLM, BertOnlyMLMHead, BertConfig, BertEmbeddings, BertPooler
from custom_bert_encoder import CustomBertEncoder  
from custom_bert_layer import CustomBertLayer  


class CustomBertModel(BertModel):
    def __init__(self, config: BertConfig, add_pooling_layer=True, alpha_attention: float = 1.0, alpha_ffn: float = 1.0):
        super().__init__(config, add_pooling_layer=add_pooling_layer) # Calls original BertModel init
        self.encoder = CustomBertEncoder(config, alpha_attention=alpha_attention, alpha_ffn=alpha_ffn)
        self.post_init()

class CustomBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config: BertConfig, alpha_attention: float = 1.0, alpha_ffn: float = 1.0):
        super().__init__(config) 
        self.bert = CustomBertModel(config, add_pooling_layer=False, alpha_attention=alpha_attention, alpha_ffn=alpha_ffn)
        self.cls = BertOnlyMLMHead(config)
        self.post_init()


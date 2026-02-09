from transformers.models.bert.modeling_bert import BertLayer, BertIntermediate, BertConfig
from custom_bert_layers import CustomBertSelfOutput, CustomBertOutput, CustomBertAttention


class CustomBertLayer(BertLayer):
    def __init__(self, config: BertConfig, alpha_attention: float = 1.0, alpha_ffn: float = 1.0):
        super(BertLayer, self).__init__() # Call nn.Module.__init__
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.attention = CustomBertAttention(config)
        self.attention.output = CustomBertSelfOutput(config, alpha=alpha_attention)

        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = CustomBertAttention(config, position_embedding_type="absolute")
            self.crossattention.output = CustomBertSelfOutput(config, alpha=alpha_attention) 

        self.intermediate = BertIntermediate(config)
        self.output = CustomBertOutput(config, alpha=alpha_ffn)


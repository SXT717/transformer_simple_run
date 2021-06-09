
import torch
from torch import nn
import torch.nn.functional as F
from transformers import RobertaTokenizer
import numpy as np

n_heads = 12
d_q = d_k = d_v = 64

class BartLearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(past_key_values_length, past_key_values_length + seq_len, dtype=torch.long)
        return super().forward(positions + self.offset)

class BartAttention(nn.Module):
    def __init__(self, embed_dim, bias=True):
        super(BartAttention, self).__init__()
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x, k_v=None):
        is_cross_attention = k_v is not None
        batch_size = x.size(0)
        q = self.q_proj(x).view(batch_size, -1, n_heads, d_q).transpose(1,2)    #[batch_size, n_heads, seq_len, d_q]
        if is_cross_attention: #
            k = self.k_proj(k_v).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
            v = self.v_proj(k_v).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_v]
        else:
            k = self.k_proj(x).view(batch_size, -1, n_heads, d_k).transpose(1,2)    #[batch_size, n_heads, seq_len, d_k]
            v = self.v_proj(x).view(batch_size, -1, n_heads, d_v).transpose(1,2)    #[batch_size, n_heads, seq_len, d_v]
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        y = self.out_proj(context)
        return y

class BartEncoderLayer(nn.Module):
    def __init__(self, embed_dim, encoder_ffn_dim):
        super(BartEncoderLayer, self).__init__()
        self.self_attn = BartAttention(embed_dim=embed_dim)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, encoder_ffn_dim)
        self.fc2 = nn.Linear(encoder_ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        x = self.self_attn(x)
        x = self.self_attn_layer_norm(x + residual)
        residual = x
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        y = self.final_layer_norm(x + residual)
        return y

class BartDecoderLayer(nn.Module):
    def __init__(self, embed_dim, decoder_ffn_dim):
        super(BartDecoderLayer, self).__init__()
        self.self_attn = BartAttention(embed_dim=embed_dim)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.encoder_attn = BartAttention(embed_dim)
        self.encoder_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, decoder_ffn_dim)
        self.fc2 = nn.Linear(decoder_ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, encoder_y):
        residual = x
        x = self.self_attn(x)
        x = self.self_attn_layer_norm(residual + x)
        residual = x
        x = self.encoder_attn(x, encoder_y)
        x = self.encoder_attn_layer_norm(residual + x)
        residual = x
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        y = self.final_layer_norm(residual + x)
        return y

class BartEncoder(nn.Module):
    def __init__(self, shard_embed, pos_dim, embed_dim, encoder_ffn_dim, encoder_num):
        super(BartEncoder, self).__init__()
        self.embed_tokens = shard_embed
        self.embed_positions = BartLearnedPositionalEmbedding(pos_dim, embed_dim)
        self.layers = nn.ModuleList([BartEncoderLayer(embed_dim, encoder_ffn_dim) for _ in range(encoder_num)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.layernorm_embedding(self.embed_tokens(x) + self.embed_positions(x.shape))
        for layer in self.layers: x = layer(x)
        return x

class BartDecoder(nn.Module):
    def __init__(self, shard_embed, pos_dim, embed_dim, decoder_ffn_dim, decoder_num):
        super(BartDecoder, self).__init__()
        self.embed_tokens = shard_embed
        self.embed_positions = BartLearnedPositionalEmbedding(pos_dim, embed_dim)
        self.layers = nn.ModuleList([BartDecoderLayer(embed_dim, decoder_ffn_dim) for _ in range(decoder_num)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)
    def forward(self, x, encoder_y):
        x = self.layernorm_embedding(self.embed_tokens(x) + self.embed_positions(x.shape))
        for layer in self.layers: x = layer(x, encoder_y)
        return x

class BartModel(nn.Module):
    def __init__(self, word_dim, pos_dim, embed_dim,
                 encoder_ffn_dim, decoder_ffn_dim, encoder_num, decoder_num):
        super(BartModel, self).__init__()
        self.decoder_start_token_id = 2 #起始符号
        self.padding_idx = 1            #填充符号
        self.shared = nn.Embedding(word_dim, embed_dim, padding_idx=self.padding_idx)
        self.encoder = BartEncoder(self.shared, pos_dim, embed_dim, encoder_ffn_dim, encoder_num)
        self.decoder = BartDecoder(self.shared, pos_dim, embed_dim, decoder_ffn_dim, decoder_num)

class BartForConditionalGeneration(nn.Module):
    def __init__(self, word_dim=50265, pos_dim=1024, embed_dim=n_heads*d_q,
                 encoder_ffn_dim=3072, decoder_ffn_dim=3072, encoder_num=6, decoder_num=6):
        super(BartForConditionalGeneration, self).__init__()
        self.final_logits_bias = nn.Parameter(torch.zeros(size=[1, word_dim]))
        self.model = BartModel(word_dim, pos_dim, embed_dim, encoder_ffn_dim, decoder_ffn_dim, encoder_num, decoder_num)
        self.lm_head = nn.Linear(embed_dim, word_dim, bias=False)

    def generate(self, encoder_x, max_length=128):
        encoder_y = self.model.encoder(encoder_x)
        decoder_x = torch.ones(size=[1, 1], dtype=torch.int64) * self.model.decoder_start_token_id
        for i in range(max_length):
            decoder_y = self.model.decoder(decoder_x, encoder_y)
            lm_logits = self.lm_head(decoder_y) + self.final_logits_bias
            new_token = torch.argmax(lm_logits[:, -1, :])
            decoder_x = torch.cat([decoder_x, torch.LongTensor([[new_token]])], dim=-1)
            if new_token.item() == self.model.decoder_start_token_id: break
        return decoder_x[0, :]

if __name__ == '__main__':
    tokenizer = RobertaTokenizer.from_pretrained('vocab/')
    examples = ['Are you chinese?']
    inputs = tokenizer(examples, padding=True, return_tensors="pt")['input_ids']
    model = BartForConditionalGeneration()
    model.load_state_dict(torch.load('model.dict'))
    outputs = model.generate(inputs)
    print('inputs = ', inputs, 'outputs = ', outputs)
    print('---', tokenizer.decode(outputs, skip_special_tokens=True))
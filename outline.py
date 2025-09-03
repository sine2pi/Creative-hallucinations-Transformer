
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class AudioEncoder(nn.Module):
    def __init__(self, in_features=128, out_features=512):
        super().__init__()

        self.conv_stack = nn.Sequential(
            nn.Conv1d(in_features, 256, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(256, 512, 5, stride=2, padding=2), nn.ReLU())
        self.proj = nn.Linear(512, out_features)

    def forward(self, x):
        x = self.conv_stack(x)
        x = x.permute(0, 2, 1) 
        return self.proj(x)

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        return self.transformer_encoder(x)

class AttentionA(nn.Module):
    def __init__(self, dims: int, head: int):
        super().__init__()
        self.head = head
        self.dims = dims
        self.head_dim = dims // head
        
        self.ln = nn.LayerNorm(dims)
        self.q = nn.Linear(dims, dims, bias=False)
        self.kv = nn.Linear(dims, dims * 2, bias=False)
        self.out = nn.Linear(dims, dims, bias=False)
        
        self.x_conv = nn.Conv2d(head, head, 1, bias=False)
        self.xa_conv = nn.Conv2d(head, head, 1, bias=False)

    def forward(self, x, xa, mask=None):

        b, n, d = x.shape
        b, m, d = xa.shape

        q = self.q(self.ln(x))
        k, v = self.kv(self.ln(x)).chunk(2, dim=-1)
        ka, va = self.kv(self.ln(xa)).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.head), (q, k, v))
        ka, va = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.head), (ka, va))

        # Text attends to audio
        attn_weights_x = torch.einsum('b h q d, b h k d -> b h q k', q, ka)
        attn_probs_x = F.softmax(attn_weights_x, dim=-1)
        x_updated = torch.einsum('b h q k, b h k d -> b h q d', attn_probs_x, va)
        
        # Audio attends to text (bidirectional)
        attn_weights_xa = torch.einsum('b h k d, b h q d -> b h k q', ka, q)
        attn_probs_xa = F.softmax(attn_weights_xa, dim=-1)
        xa_updated = torch.einsum('b h k q, b h q d -> b h k d', attn_probs_xa, v)

        x_updated = rearrange(x_updated, 'b h q d -> b q (h d)')
        xa_updated = rearrange(xa_updated, 'b h k d -> b k (h d)')

        return self.out(x_updated), self.out(xa_updated)

class CreativeFusionBlock(nn.Module):
    def __init__(self, dims: int, head: int, blend: bool = True, modal: bool = True):
        super().__init__()
        self.blend = blend
        self.modal = modal

        self.intra_text = nn.TransformerEncoderLayer(dims, head, batch_first=True)
        self.intra_audio = nn.TransformerEncoderLayer(dims, head, batch_first=True)

        self.cross_attn = AttentionA(dims, head)
   
        if self.modal:
            self.joint_attn = nn.TransformerEncoderLayer(dims * 2, head, batch_first=True)
            
        if self.blend:
            self.register_parameter('blend_weight', nn.Parameter(torch.zeros(1)))

    def forward(self, x, xa, mask=None):
  
        y = x.clone()
        x = self.intra_text(x, src_key_padding_mask=mask)
        xa = self.intra_audio(xa)

        x_fused, xa_fused = self.cross_attn(x, xa, mask=mask)
        x = x + x_fused
        xa = xa + xa_fused

        if self.blend:
            alpha = torch.sigmoid(self.blend_weight)
            x = alpha * x + (1 - alpha) * y

        if self.modal:
            xm = self.joint_attn(torch.cat([x, xa], dim=-1))
            x = xm[:, :x.shape[1]]
            xa = xm[:, x.shape[1]:]

        return x, xa

class CreativeFusionTransformer(nn.Module):
    def __init__(self, vocab_size, dims=512, head=8, num_layers=6, max_seq_len=100):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.audio_encoder = AudioEncoder(out_features=dims)
        self.text_encoder = TextEncoder(vocab_size, embed_dim=dims)

        self.blocks = nn.ModuleList([CreativeFusionBlock(dims, head) for _ in range(num_layers)])

        self.decoder_emb = nn.Embedding(vocab_size, dims)
        self.decoder_transformer = nn.TransformerDecoderLayer(dims, head, batch_first=True)
        self.decoder_head = nn.Linear(dims, vocab_size)

    def get_embeddings(self, audio_spec, text_tokens):
        xa = self.audio_encoder(audio_spec)
        x = self.text_encoder(text_tokens)
        audio_emb = torch.mean(xa, dim=1)
        text_emb = torch.mean(x, dim=1)
        return audio_emb, text_emb

    def forward(self, audio_spec, text_tokens):

        xa = self.audio_encoder(audio_spec)
        x = self.text_encoder(text_tokens)
        
        for block in self.blocks:
            x, xa = block(x, xa)

        decoder_input = self.decoder_emb(text_tokens)
        memory = torch.cat([x, xa], dim=1) # Combine fused representations
        output = self.decoder_transformer(decoder_input, memory)
        logits = self.decoder_head(output)
        return logits
    
    def generate(self, audio_spec, start_token_id, end_token_id, max_gen_len=50):
        self.eval()
        with torch.no_grad():
            xa = self.audio_encoder(audio_spec)
            x = torch.zeros(audio_spec.shape[0], 1, self.decoder_emb.embedding_dim, device=audio_spec.device)

            for block in self.blocks:
                x_fused, xa_fused = block(x, xa)
            
            memory = torch.cat([x_fused, xa_fused], dim=1)

            generated_tokens = torch.full((audio_spec.shape[0], 1), start_token_id, dtype=torch.long, device=audio_spec.device)

            for _ in range(max_gen_len):
                decoder_input = self.decoder_emb(generated_tokens)
                output = self.decoder_transformer(decoder_input, memory)
                logits = self.decoder_head(output[:, -1, :])
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
                generated_tokens = torch.cat([generated_tokens, next_token_id], dim=1)
                
                if (next_token_id == end_token_id).all():
                    break

            return generated_tokens

class InfoNCELoss(nn.Module):

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(self, audio_embeddings: torch.Tensor, text_embeddings: torch.Tensor):

        audio_embeddings = F.normalize(audio_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        sim_matrix = torch.matmul(audio_embeddings, text_embeddings.T) / self.temperature
        labels = torch.arange(len(sim_matrix), device=sim_matrix.device)
    
        loss_a_to_t = self.cross_entropy(sim_matrix, labels)
        loss_t_to_a = self.cross_entropy(sim_matrix.T, labels)
        return (loss_a_to_t + loss_t_to_a) / 2

import torch
import torch.nn as nn
import torch
import torch.nn as nn
from src.seca import ScalarExpansionContractiveAutoencoder
from src.layers import PositionalEmbeddingLayer
from src.modules import EncoderModule, DecoderModule, Output
from typing import Optional, List


class TransformerLikeModel(nn.Module):
  def __init__(self, embed_size: int, encoder_size: int = 6, decoder_size: int = 6, input_size: int = 1, hidden_ff_size_enc: Optional[int] = None, hidden_ff_size_dec: Optional[int] = None, num_head_enc: int = 8, num_head_dec_1: int = 8, num_head_dec_2: int = 8,
                positional_embedding_method: str = "learnable", max_seq_length: int = 120, cls_token_method: str = "learnable", output_len: int = 6, seca: Optional[ScalarExpansionContractiveAutoencoder] = None, dropout: float = 0.0):
    super(TransformerLikeModel, self).__init__()

    self.embed_size = embed_size
    self.input_size = input_size
    self.hidden_ff_size_enc = hidden_ff_size_enc if hidden_ff_size_enc is not None else embed_size * 4
    self.hidden_ff_size_dec = hidden_ff_size_dec if hidden_ff_size_dec is not None else embed_size * 4

    self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size)) if cls_token_method == 'learnable' else nn.Parameter(torch.ones(1, 1, embed_size), requires_grad=False)
    self.output_len = output_len

    self.seca = ScalarExpansionContractiveAutoencoder(embed_size, input_size) if seca is None else seca
    self.pe = PositionalEmbeddingLayer(max(max_seq_length, output_len), embed_size, positional_embedding_method)
    self.encoder = nn.Sequential(*[
        EncoderModule(embed_size, num_head_enc, self.hidden_ff_size_enc, dropout=dropout) for _ in range(encoder_size)
    ])
    self.decoder = nn.Sequential(*[
        DecoderModule(embed_size, num_head_dec_1, num_head_dec_2, self.hidden_ff_size_dec, dropout=dropout) for _ in range(decoder_size)
    ])
    self.output = Output(embed_size, hidden_dim=embed_size*2)

  def single_forward(self, input: tuple[torch.Tensor, torch.Tensor]):
    """From the raw input X and the already generated Y, get next element (UNDECODED).

    Returns the raw decoded prediction token (still in embed space) — Output handles final scaling/bias.
    """
    X, Y = input

    Z = self.seca.encode(X)
    Z = self.pe(Z)
    Z = self.encoder(Z)

    Y = self.pe(Y)
    y = self.decoder((Y, Z))[0]
    context = Z.mean(dim=1)
    y = self.output(y, context=context)

    return y

  def forward(self, X: torch.Tensor):
    """Autoregressive forward: returns decoded absolute predictions shaped (batch, output_len, 1).

    The decoder tokens stored in Y_tokens are raw embeddings; positional embeddings are applied
    only when feeding into the decoder per step.
    """
    batch_size, seq_length, _ = X.shape
    Y_tokens = self.cls_token.expand((batch_size, 1, self.embed_size))

    Z = self.seca.encode(X)
    Z = self.pe(Z)
    Z = self.encoder(Z)

    preds = []

    for _ in range(self.output_len):
      Y_pe = self.pe(Y_tokens)
      y = self.decoder((Y_pe, Z))[0]
      context = Z.mean(dim=1)
      y = self.output(y, context=context)
      Y_tokens = torch.cat([Y_tokens, y.unsqueeze(1)], dim=1)
      preds.append(self.seca.decode(y))

    return torch.stack(preds).permute(1, 0, 2)

  def save_model(self, path: str):
    torch.save(self.state_dict(), path)

  @staticmethod
  def load_model(path: str, model_class, *args, **kwargs):
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    return model

  def get_encoder_attention(self, X):
    Z = self.seca.encode(X)
    Z = self.pe(Z)
    attn_layer = self.encoder[0].mha
    _, attn_weights = attn_layer((Z, Z, Z), return_attention=True) # type: ignore
    return attn_weights

  def get_cross_attention(self, X, Y):
    Z = self.seca.encode(X)
    Z = self.pe(Z)
    Z = self.encoder(Z)

    Y = self.seca.encode(Y)
    Y = self.pe(Y)
    Y = self.decoder[0].mha_1((Y, Y, Y)) # type: ignore
    Y = self.decoder[0].norm1(Y) # type: ignore
    _, attention_weights = self.decoder[0].mha_2((Y, Z, Z), return_attention=True) # type: ignore

    return attention_weights


class EncoderOnlyModel(nn.Module):
  def __init__(self, embed_size: int, encoder_size: int = 6, input_size: int = 1, output_len: int = 1, hidden_ff_size_enc: Optional[int] = None, num_head_enc: int = 8, positional_embedding_method: str = "fixed", max_seq_length: int = 120):
    super(EncoderOnlyModel, self).__init__()

    self.embed_size = embed_size
    self.input_size = input_size
    self.hidden_ff_size_enc = hidden_ff_size_enc if hidden_ff_size_enc is not None else embed_size * 4
    self.output_len = output_len

    self.seca = ScalarExpansionContractiveAutoencoder(embed_size, input_size)
    self.pe = PositionalEmbeddingLayer(max_seq_length, embed_size, positional_embedding_method)
    self.encoder = nn.Sequential(*[
        EncoderModule(embed_size, num_head_enc, self.hidden_ff_size_enc) for _ in range(encoder_size)
    ])
    self.output = Output(embed_size)

  def forward(self, X: torch.Tensor) -> torch.Tensor:
    preds = []

    for _ in range(self.output_len):
      Z = self.seca.encode(X)
      Z = self.pe(Z)
      Z = self.encoder(Z)
      context = Z.mean(dim=1)
      output = self.output(Z, context=context)
      decoded_output = self.seca.decode(output)
      X = torch.cat(
        (X[:, 1:], decoded_output), dim=1
      )
      preds.append(decoded_output.squeeze(1))

    return torch.stack(preds, dim=1)

  def single_forward(self, input: torch.Tensor):
    Z = self.seca.encode(input)
    Z = self.pe(Z)
    Z = self.encoder(Z)
    context = Z.mean(dim=1)
    output = self.output(Z, context=context)
    return output
    Z = self.seca.encode(input)
    Z = self.pe(Z)
    Z = self.encoder(Z)
    output = self.output(Z)
    return output
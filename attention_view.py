"""Visualize cross-attention per decoded step for a single-decoder TransformerLikeModel.

Usage (example):
  python attention_view.py --model-path ristoranti_model.pth --embed-size 64

This script loads a model, prepares an input sequence (or loads from a .npy), runs the autoregressive
decoding step-by-step and captures the cross-attention used to produce each output token. It then
plots an attention-over-time heatmap (query=decode step, key=encoder positions) using the
helpers in `attention_plot_examples.py`.

Assumes a single decoder layer (the repository's model uses sequential decoder modules; this script
captures attention from `model.decoder[0]`).
"""

import argparse
import os
from typing import Optional, Any

import numpy as np
import pandas as pd
import torch

from src.model import TransformerLikeModel
from attention_plot import plot_cross_attention, topk_alignments, plot_weighted_input_contributions, plot_weighted_input_grid, plot_series_with_mixed_attention_grid

def prepare_input(input_npy: np.ndarray, seq_len: int, input_size: int, device: str = 'cpu'):

    x = input_npy
    if x.ndim == 1:
        x = x.reshape(1, -1, 1)
    elif x.ndim == 2:
        x = x[np.newaxis, :, :]
    x = x.astype(np.float32)
    return torch.from_numpy(x).to(device)


def capture_cross_attention_per_step(model: TransformerLikeModel, X: torch.Tensor, device: str = 'cpu'):
    """Capture cross-attention used to generate each autoregressive output step.

    Returns:
      att_steps: numpy array shaped (steps, batch, heads, k_len)
      X: original input tensor
    """
    # Put model on device
    model.to(device)
    model.eval()

    batch = X.shape[0]

    # Encode input
    with torch.no_grad():
        Z = model.seca.encode(X)
        if model.use_pe:
            Z = model.pe(Z)
        Z_enc = model.encoder(Z)

    Y_tokens = model.cls_token.expand((batch, 1, model.embed_size)).to(device)

    steps = model.output_len
    att_list = []
    self_att_list = []
    predictions = []

    for step in range(steps):
        with torch.no_grad():
            Y_in = Y_tokens
            if model.use_pe:
                Y_pe = model.pe(Y_in)
            else:
                Y_pe = Y_in
            mha1: Any = model.decoder[0].mha_1
            mha2: Any = model.decoder[0].mha_2
            dropout: Any = model.decoder[0].dropout
            mha1_out, self_w = mha1((Y_pe, Y_pe, Y_pe), return_attention=True)
            norm1: Any = getattr(model.decoder[0], 'norm1')
            Y_normed = norm1(Y_pe + dropout(mha1_out))

            # capture self-attention last-query over decoder positions (batch, heads, dec_len)
            # mha1 returns attention weights typically shaped (batch, heads, dec_len, dec_len).
            # We only need the last query's distribution over decoder positions for the
            # autoregressive decoding step (i.e. attention of the newest token over previous tokens).
            try:
                self_np_full = self_w.detach().cpu().numpy()
            except Exception:
                # fallback if self_w is already a numpy array
                self_np_full = np.array(self_w)
            # Store the full self-attention tensor when available (batch, heads, dec_len, dec_len)
            # Plotting utilities will extract last-query rows or average heads as needed.
            self_att_list.append(self_np_full)

            _, cross_w = mha2((Y_normed, Z_enc, Z_enc), return_attention=True)

            last_q_cross = cross_w[:, :, -1, :].detach().cpu().numpy()
            att_list.append(last_q_cross)

            # produce decoder output for current Y_in to obtain next-token prediction
            dec_out, _ = model.decoder((Y_pe, Z_enc))
            if model.use_out:
                context = Z_enc.mean(dim=1)
                y = model.output(dec_out, context=context)
                y_token = y
            else:
                y_token = dec_out[:, -1, :]

            # decoded scalar prediction (after seca decode) for plotting
            try:
                y_decoded = model.seca.decode(y).detach().cpu().numpy()
            except Exception:
                # fallback: if y is raw token, attempt to decode the last token
                try:
                    last_dec = dec_out[:, -1, :]
                    y_decoded = model.seca.decode(last_dec).detach().cpu().numpy()
                except Exception:
                    y_decoded = np.zeros((batch, 1))
            # append batch-0 prediction value
            predictions.append(y_decoded[0].reshape(-1)[0])

            Y_tokens = torch.cat([Y_tokens, y_token.unsqueeze(1)], dim=1)

    att_steps = np.stack(att_list, axis=0)  # (steps, batch, heads, k_len)
    # self_att_list is list of arrays (batch, heads, dec_len) per step
    return att_steps, self_att_list, np.array(predictions), X


def capture_attention_teacher_forcing(model: TransformerLikeModel, X: torch.Tensor, Y_true: torch.Tensor, device: str = 'cpu'):
    """Capture attention when progressively revealing the true Y tokens (teacher-forcing style).

    For each run i (1..len(Y_true)), build Y_input = [CLS] + Y_true[:i] and run decoder once to
    get cross-attention and self-attention for that run. Returns lists of cross- and self-attention
    per run, plus the predictions array (decoded model outputs for each run) and X.
    """
    model.to(device)
    model.eval()

    batch = X.shape[0]

    # Encode input once
    with torch.no_grad():
        Z = model.seca.encode(X)
        if model.use_pe:
            Z = model.pe(Z)
        Z_enc = model.encoder(Z)

    # Prepare true token embeddings for Y_true (assume Y_true are raw scalars)
    # Convert Y_true scalars to token embeddings via seca.encode
    with torch.no_grad():
        Y_tokens_true = model.seca.encode(Y_true.to(device))
        if model.use_pe:
            Y_tokens_true = model.pe(Y_tokens_true)

    runs_cross = []
    runs_self = []
    run_preds = []

    cls = model.cls_token.expand((batch, 1, model.embed_size)).to(device)

    # For each progressive length i (1..Y_len), build Y_input and capture attentions
    Y_len = Y_tokens_true.shape[1]
    for i in range(1, Y_len + 1):
        with torch.no_grad():
            # Y_input: CLS followed by first i true tokens (without re-applying pe here since we pre-pe'd)
            Y_in = torch.cat([cls, Y_tokens_true[:, :i, :]], dim=1)
            # If model expects pe inside decoder, ensure we feed pe'd tokens -- above we pe'd already
            # but the decoder in this repository applies pe before MHA, so safe.

            mha1: Any = model.decoder[0].mha_1
            mha2: Any = model.decoder[0].mha_2
            dropout: Any = model.decoder[0].dropout
            mha1_out, self_w = mha1((Y_in, Y_in, Y_in), return_attention=True)
            norm1: Any = getattr(model.decoder[0], 'norm1')
            Y_normed = norm1(Y_in + dropout(mha1_out))

            # process self-att: take last query when full matrix present
            try:
                self_np_full = self_w.detach().cpu().numpy()
            except Exception:
                self_np_full = np.array(self_w)
            if getattr(self_np_full, 'ndim', None) == 4:
                last_q_self = self_np_full[:, :, -1, :]
            else:
                last_q_self = self_np_full

            _, cross_w = mha2((Y_normed, Z_enc, Z_enc), return_attention=True)

            last_q_cross = cross_w[:, :, -1, :].detach().cpu().numpy()

            # append
            runs_cross.append(last_q_cross)
            runs_self.append(last_q_self)

            # run decoder full forward for this Y_in to get model's prediction for next token
            dec_out, _ = model.decoder((Y_in, Z_enc))
            if model.use_out:
                context = Z_enc.mean(dim=1)
                y = model.output(dec_out, context=context)
                y_token = y[:, -1, :]
            else:
                y_token = dec_out[:, -1, :]

            try:
                y_decoded = model.seca.decode(y_token).detach().cpu().numpy()
            except Exception:
                y_decoded = np.zeros((batch, 1))
            run_preds.append(y_decoded[0].reshape(-1)[0])

    # Stack cross att across runs: resulting shape (runs, batch, heads, k_len)
    cross_arr = np.stack(runs_cross, axis=0)
    # self att kept as list per run (each element (batch, heads, dec_len))
    return cross_arr, runs_self, np.array(run_preds), X

SEQUENCE_LENGTH = 7
PREDICTION_LENGTH = 7
EMBED_SIZE = 4
ENCODER_SIZE = 1
DECODER_SIZE = 1
NUM_HEADS = 2
BATCH_SIZE = 2
EPOCHS = 500
DROPOUT = 0.05
TRAIN_PERCENTAGE = 0.75
SEED = 42
DELTA = False
EARLY_STOPPING = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed-size', type=int, default=EMBED_SIZE, help='Embed size used when constructing the model')
    parser.add_argument('--seq-len', type=int, default=SEQUENCE_LENGTH)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--show-topk', type=int, default=3, help='Show top-k key positions per step')
    parser.add_argument('--teacher-forcing', action='store_true', help='Run teacher-forcing progressive Y reveals and plot per-run attention')
    args = parser.parse_args()

    device = args.device

    print('Loading model...')
    model: TransformerLikeModel = TransformerLikeModel.load_model("ristoranti_model.pth", TransformerLikeModel, embed_size=EMBED_SIZE,
        encoder_size=ENCODER_SIZE,
        decoder_size=DECODER_SIZE,
        num_head_dec_1=NUM_HEADS,
        num_head_dec_2=NUM_HEADS,
        num_head_enc=NUM_HEADS,
        output_len=PREDICTION_LENGTH,
        max_seq_length=SEQUENCE_LENGTH,
    )

    df = pd.read_csv('ristorantiGTrend.csv')
    series = df.iloc[:, 1].values.astype(np.float32)

    series = (series - np.min(series)) / (np.max(series) - np.min(series))
    input_npy = series[:SEQUENCE_LENGTH]
    X = prepare_input(input_npy, seq_len=args.seq_len, input_size=model.input_size, device=device)

    att_steps, self_att_list, predictions, X = capture_cross_attention_per_step(model, X, device=device)

    steps, batch, heads, k_len = att_steps.shape

    # prepare cross-att for plotting: (heads, steps, k_len)
    att_for_plot = att_steps[:, 0, :, :]  # (steps, heads, k_len)
    att_for_plot = np.transpose(att_for_plot, (1, 0, 2))

    # prepare self-att list: take batch-0 and convert to (heads, dec_len) per step
    self_att_proc = []
    for s in range(len(self_att_list)):
        arr = np.array(self_att_list[s])
        # arr expected shape (batch, heads, dec_len) or (heads, dec_len)
        if arr.ndim == 3:
            self_att_proc.append(arr[0])
        else:
            self_att_proc.append(arr)

    input_series = X[0, :, 0].detach().cpu().numpy() if isinstance(X, torch.Tensor) and X.ndim == 3 else None

    if input_series is not None:
        print('\nPlotting series + mixed attention per predicted step (grid)...')
        try:
            plot_series_with_mixed_attention_grid(att_for_plot, self_att_proc, input_series=input_series, predictions=predictions, agg_mode='mean', figsize=(12, 2), font_size=12)
        except Exception as e:
            print(f'Failed plotting mixed-attention grid: {e}')
    else:
        print('Input series unavailable; skipping mixed-attention plots.')

    # If teacher-forcing mode requested, progressively reveal true Y and plot per-run attention
    if args.teacher_forcing:
        print('\nRunning teacher-forcing progressive attention capture...')
        # prepare true future Y tokens from the series immediately after the X window
        start_idx = 16 + SEQUENCE_LENGTH
        Y_true_vals = series[start_idx:start_idx + PREDICTION_LENGTH]
        # shape (batch, Y_len, 1)
        Y_true = Y_true_vals.reshape(1, -1, 1).astype(np.float32)
        Y_t = torch.from_numpy(Y_true).to(device)

        cross_runs, self_runs, run_preds, _ = capture_attention_teacher_forcing(model, X, Y_t, device=device)

        # cross_runs shape: (runs, batch, heads, k_len) -> convert to (heads, runs, k_len) for plotting
        runs, batch, heads, k_len = cross_runs.shape
        cross_for_plot = cross_runs[:, 0, :, :]  # (runs, heads, k_len)
        cross_for_plot = np.transpose(cross_for_plot, (1, 0, 2))  # (heads, runs, k_len)

        # prepare self_runs list: take batch-0 and ensure shape (heads, dec_len) per run
        self_runs_proc = []
        for s in range(len(self_runs)):
            arr = np.array(self_runs[s])
            if arr.ndim == 3:
                self_runs_proc.append(arr[0])
            else:
                self_runs_proc.append(arr)

        print('Plotting teacher-forcing per-run attention (cross and self)...')
        try:
            if input_series is None:
                # fallback: create a zeros input series of appropriate length for plotting
                input_series_plot = np.zeros(k_len)
            else:
                input_series_plot = input_series
            plot_series_with_mixed_attention_grid(cross_for_plot, self_runs_proc, input_series=input_series_plot, predictions=run_preds, agg_mode='mean', figsize=(12, 2), font_size=12, show_self_att=True)
        except Exception as e:
            print(f'Failed plotting teacher-forcing attention grid: {e}')


if __name__ == '__main__':
    main()

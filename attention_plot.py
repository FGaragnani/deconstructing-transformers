"""Small utilities to extract and visualize attention from the provided Transformer-like model.

Usage examples:
    from attention_plot_examples import plot_cross_attention
    att = model.get_cross_attention(X_tensor, Y_tensor)  # att: (batch, heads, q_len, k_len)
    plot_cross_attention(att[0])  # plot first batch element

This file provides:
 - plot_cross_attention: heatmaps per head + aggregated heatmap
 - aggregate_attention: mean or max across heads
 - topk_alignments: returns top-k key indices per query position

The plotting functions use matplotlib only (already in requirements.txt).
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def aggregate_attention(att: np.ndarray, mode: str = 'mean') -> np.ndarray:
    """Aggregate heads into a single attention map.

    Args:
        att: (heads, q_len, k_len) or (batch, heads, q_len, k_len)
        mode: 'mean' or 'max'
    Returns:
        agg: (q_len, k_len)
    """
    # Support an optional leading "layers" dimension when there is a single decoder layer
    # Expected final shape: (heads, q_len, k_len) or (batch, heads, q_len, k_len)
    if att.ndim == 5:
        # assume shape (layers=1, batch, heads, q_len, k_len) or (layers=1, heads, q_len, k_len,?)
        # squeeze the single layer dimension
        att = np.squeeze(att, axis=0)
    if att.ndim == 4:
        att = att[0]
    if mode == 'mean':
        return att.mean(axis=0)
    elif mode == 'max':
        return att.max(axis=0)
    else:
        raise ValueError('mode must be mean or max')


def _stable_softmax(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Numerically stable softmax over the last axis for 1D arrays.

    Returns a float64 array that sums to 1, unless it already sums to almost 1.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    if np.isclose(np.sum(x), 1.0, atol=eps):
        return x
    x = x - np.max(x)
    ex = np.exp(x)
    s = ex / (np.sum(ex) + eps)
    return s


def plot_cross_attention(att: np.ndarray, input_series: Optional[np.ndarray] = None, head: Optional[int] = None, agg_mode: str = 'mean', figsize: Tuple[int, int] = (12, 4)):
    """Plot attention heatmaps for cross-attention.

    Assumes a single decoder layer (if a leading "layers" dimension is present it will be squeezed).

    att: (heads, q_len, k_len) or (batch, heads, q_len, k_len) or optionally (layers=1, batch, heads, q_len, k_len)
    If head is None: plot all heads in a row + aggregated map. Otherwise only plot that head and aggregated.
    If input_series is provided (shape k_len or k_len x features) it will be shown below the aggregated heatmap.
    """
    if att.ndim == 5 and att.shape[0] == 1:
        att = np.squeeze(att, axis=0)
    if att.ndim == 4:
        att = att[0]

    heads, q_len, k_len = att.shape

    agg = aggregate_attention(att, mode=agg_mode)

    if head is None:
        ncols = min(heads, 8) + 1  # show up to 8 heads then aggregate
        fig, axes = plt.subplots(1, ncols, figsize=(figsize[0], 2.0 * ncols))
        for i in range(ncols - 1):
            ax = axes[i]
            im = ax.imshow(att[i], aspect='auto', origin='lower')
            ax.set_title(f'Head {i}')
            ax.set_xlabel('Key positions (encoder)')
            ax.set_ylabel('Query positions (decoder)')
            fig.colorbar(im, ax=ax)
        ax = axes[-1]
        im = ax.imshow(agg, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title('Aggregated')
        fig.colorbar(im, ax=ax)
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        im = axes[0].imshow(att[head], aspect='auto', origin='lower')
        axes[0].set_title(f'Head {head}')
        axes[0].set_xlabel('Key positions (encoder)')
        axes[0].set_ylabel('Query positions (decoder)')
        fig.colorbar(im, ax=axes[0])
        im2 = axes[1].imshow(agg, aspect='auto', origin='lower', cmap='viridis')
        axes[1].set_title('Aggregated')
        fig.colorbar(im2, ax=axes[1])

    if input_series is not None:
        # overlay the input series (k_len) under the aggregated map for reference
        plt.figure(figsize=(figsize[0], 2))
        plt.plot(np.arange(k_len), input_series, '-o')
        plt.title('Encoder input series (key positions)')
        plt.xlabel('Position')
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def topk_alignments(att: np.ndarray, k: int = 3) -> np.ndarray:
    """Return top-k key indices per query for aggregated attention.

    Returns: (q_len, k) int indices of key positions
    """
    # Ensure single-decoder-layer shapes are handled by aggregate_attention
    agg = aggregate_attention(att, mode='mean')
    # each row is a distribution over keys for a query position
    topk = np.argsort(-agg, axis=-1)[:, :k]
    return topk


def plot_weighted_input_contributions(att: np.ndarray, input_series: np.ndarray, step: Optional[int] = None, agg_mode: str = 'mean', figsize: Tuple[int, int] = (8, 4), font_size: int = 16):
    """Plot the encoder input series where marker sizes reflect the attention contribution to a predicted step.

    Args:
        att: attention array shaped (heads, q_len, k_len) or (batch, heads, q_len, k_len)
        input_series: 1D array of length k_len (encoder positions values)
        step: which decode step (0-based). If None and q_len==1, uses that single query. If None and q_len>1, raises.
        agg_mode: how to aggregate heads ('mean' or 'max').
    """
    if att.ndim == 4:
        att = att[0]

    heads, q_len, k_len = att.shape
    if step is None:
        if q_len == 1:
            step = 0
        else:
            raise ValueError('When q_len>1 you must provide a step index')

    # aggregate heads first
    agg = aggregate_attention(att, mode=agg_mode)  # (q_len, k_len)
    # get attention vector for requested step
    att_vec = agg[step]  # (k_len,)

    # normalize sizes for plotting (min marker size 20, max 400)
    eps = 1e-8
    norm = (att_vec - att_vec.min()) / (att_vec.max() - att_vec.min() + eps)
    sizes = 20 + norm * 380

    # set font size temporarily
    prev_fs = plt.rcParams.get('font.size', None)
    plt.rcParams.update({'font.size': font_size})

    plt.figure(figsize=figsize)
    plt.plot(np.arange(k_len), input_series, '-o', markersize=6, label='Input series')
    plt.scatter(np.arange(k_len), input_series, s=sizes, c='C1', alpha=0.6, label='Attention contribution')
    plt.title(f'Input contributions to predicted step {step}', fontsize=16)
    plt.xlabel('Encoder position')
    plt.ylabel('Input value')
    plt.legend()
    plt.grid(True)
    plt.show()

    # restore previous font size
    if prev_fs is not None:
        plt.rcParams.update({'font.size': prev_fs})


def plot_weighted_input_grid(att: np.ndarray, input_series: np.ndarray, ncols: int = 2, agg_mode: str = 'mean', figsize: Tuple[int, int] = (10, 6), suptitle: Optional[str] = None, font_size: int = 12):
    """Plot weighted-input contribution charts for all decode steps arranged in a grid with ncols columns.

    Args:
        att: attention array shaped (heads, q_len, k_len) or (batch, heads, q_len, k_len)
        input_series: 1D array of length k_len
        ncols: number of columns in the grid (defaults to 2)
        agg_mode: how to aggregate heads
        figsize: overall figure size
        suptitle: optional super title for the figure
    """
    if att.ndim == 4:
        att = att[0]

    heads, q_len, k_len = att.shape
    nplots = q_len
    nrows = (nplots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    if suptitle:
        fig.suptitle(suptitle)

    # set font size temporarily
    prev_fs = plt.rcParams.get('font.size', None)
    plt.rcParams.update({'font.size': font_size})

    agg = aggregate_attention(att, mode=agg_mode)  # (q_len, k_len)

    for i in range(nplots):
        r = i // ncols
        c = i % ncols
        ax = axes[r][c]
        att_vec = agg[i]
        eps = 1e-8
        norm = (att_vec - att_vec.min()) / (att_vec.max() - att_vec.min() + eps)
        sizes = 20 + norm * 380

        ax.plot(np.arange(k_len), input_series, '-o', markersize=4, label='Input series')
        ax.scatter(np.arange(k_len), input_series, s=sizes, c='C1', alpha=0.6, label='Attention contribution')
        ax.set_title(f'Step {i+1}', fontsize=18)
        ax.set_ylabel('Value', fontsize=18)
        ax.grid(True)
        # show legend on bottom row plots
        if r == (nrows - 1):
            ax.legend()

    # Hide any empty subplots
    for j in range(nplots, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r][c].axis('off')

    plt.tight_layout()
    plt.show()

    # restore previous font size
    if prev_fs is not None:
        plt.rcParams.update({'font.size': prev_fs})



def plot_series_with_mixed_attention_grid(cross_att: np.ndarray, self_att_list, input_series: np.ndarray, predictions: np.ndarray, agg_mode: str = 'mean', figsize: Tuple[int, int] = (12, 3), font_size: int = 12, show_self_att: bool = False):
    """Plot OUTPUT_LEN rows x N columns: full series and attention visualizations.

    By default the function creates 2 columns: input+generated markers (with cross-att sizes)
    and a zoomed view showing generated points sized by self-attention-derived scalars.
    If `show_self_att=True` an extra column is added showing the full self-attention matrix
    (averaged across heads) computed at each decoding step. Each subplot shows the attention
    used to generate token i (0-based index i). The generated token itself is NOT yet appended
    to the plotted series for that panel — the visualization shows the model's attention at the
    moment of generation.

    Marker sizes for input points are derived from cross-attention for the current predicted step.
    Marker sizes for autoregressive predicted points are derived from subsequent self-attention masses that
    attend to those decoder positions.

    Args:
        cross_att: (heads, steps, k_len) or (batch, heads, steps, k_len)
        self_att_list: list of length steps with elements that may be one of:
                       - (batch, heads, dec_len, dec_len) full self-att matrices per step
                       - (batch, heads, dec_len) last-query vectors per step
                       - (heads, dec_len) or (dec_len,) fallback shapes
        input_series: 1D array length k_len
        predictions: 1D array length steps (values generated in decoding order)
        show_self_att: if True, plot per-step self-attention heatmap as an extra column
    """
    # Normalize inputs
    if isinstance(predictions, list):
        predictions = np.asarray(predictions)
    if isinstance(input_series, (list, tuple)):
        input_series = np.asarray(input_series)

    # Reduce batch dimension if present
    att = cross_att
    if att.ndim == 4:
        # (batch, heads, steps, k_len)
        att = att[0]

    # att now (heads, steps, k_len)
    if att.ndim != 3:
        raise ValueError('cross_att must be (heads, steps, k_len) or (batch, heads, steps, k_len)')

    heads, steps, k_len = att.shape

    # Prepare self attention per step as list of (heads, dec_len)
    if isinstance(self_att_list, np.ndarray):
        if self_att_list.ndim == 3 and self_att_list.shape[0] == steps:
            # (steps, batch?, heads?) ambiguous - try (steps, heads, dec_len)
            sal = [self_att_list[s] for s in range(self_att_list.shape[0])]
        elif self_att_list.ndim == 3 and self_att_list.shape[1] == heads:
            # (batch, heads, dec_len) per step flattened? fallback
            sal = [self_att_list[s] for s in range(self_att_list.shape[0])]
        else:
            # convert along first axis
            sal = [self_att_list[s] for s in range(self_att_list.shape[0])]
    else:
        sal = list(self_att_list)

    # Ensure sal length matches steps; pad with zeros if necessary
    if len(sal) < steps:
        for _ in range(len(sal), steps):
            sal.append(np.zeros((heads, 1)))

    # Aggregate cross-attention across heads -> (steps, k_len)
    cross_agg = aggregate_attention(att, mode=agg_mode)  # (steps, k_len)

    # arrays (dec_len,)
    self_agg = []
    for s in range(steps):
        arr = np.array(sal[s])
        # support full self-att matrices: (batch, heads, dec_len, dec_len)
        if arr.ndim == 4:
            # take batch 0 -> (heads, dec_len, dec_len)
            arr = arr[0]
            # average heads -> (dec_len, dec_len)
            mat = arr.mean(axis=0)
            # take the last query row (attention of last query over decoder positions)
            row = mat[-1]
            self_agg.append(_stable_softmax(row))
        elif arr.ndim == 3:
            # (batch, heads, dec_len) -> take batch 0 -> (heads, dec_len)
            arr = arr[0]
            # mean across heads then softmax over decoder positions
            self_agg.append(_stable_softmax(arr.mean(axis=0)))
        elif arr.ndim == 2:
            # (heads, dec_len)
            self_agg.append(_stable_softmax(arr.mean(axis=0)))
        elif arr.ndim == 1:
            # softmax over decoder positions
            self_agg.append(_stable_softmax(arr))
        else:
            self_agg.append(np.zeros(1))

    # plotting grid: arrange steps across 2 columns (ncols=2)
    ncols = 2 + (1 if show_self_att else 0)
    nrows = (steps + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0], figsize[1] * nrows), squeeze=False)
    prev_fs = plt.rcParams.get('font.size', None)
    plt.rcParams.update({'font.size': font_size})

    eps = 1e-8
    for i in range(steps):
        r = i // ncols
        c = i % ncols

        # series up to and including prediction i (i==0 -> zero predictions)
        preds_up = predictions[:i] if i > 0 else np.array([])
        full_series = np.concatenate([input_series, preds_up]) if preds_up.size else input_series.copy()

        # input sizes from cross-att for step i
        cross_vec = cross_agg[i]
        # ensure cross-attention vector is a probability distribution via stable softmax
        norm_cross = _stable_softmax(cross_vec)
        sizes_cross = 0 + norm_cross * 1000

        # prediction sizes: for each pred j < i
        sizes_preds = np.array([])
        if i > 0:
            pred_sizes = []
            for j in range(i):
                dec_pos = j + 1
                vals = []
                for l in range(j + 1, i + 1):
                    agg = self_agg[l]
                    if dec_pos < len(agg):
                        vals.append(agg[dec_pos])
                pred_sizes.append(np.mean(vals) if len(vals) > 0 else 0.0)
            ps = np.array(pred_sizes)
            if ps.size > 0:
                norm_ps = ps
                sizes_preds = 0 + norm_ps * 1000

        # plot into grid cell (r, c)
        ax = axes[r][c]
        x_full = np.arange(len(full_series))
        # plot input series line in color C0
        x_inputs = np.arange(len(input_series))
        ax.plot(x_inputs, input_series, '-o', markersize=4, color='C0', label='Input series')
        # inputs (cross-att) markers (larger, attention-weighted)
        ax.scatter(x_inputs, input_series, s=sizes_cross, c='C0', alpha=0.8, edgecolors='k')
        ax.set_xlabel('Time step', fontsize=font_size)
        ax.set_ylabel('Standardized Value')

        # predictions (self-att): plot line in C2 and attention-weighted markers
        if sizes_preds.size:
            x_preds = np.arange(len(input_series), len(input_series) + len(sizes_preds))
            preds_vals = full_series[len(input_series):]
            ax.plot(x_preds, preds_vals, '-o', markersize=4, color='C2', label='Generated series')
            ax.scatter(x_preds, preds_vals, s=sizes_preds, c='C2', alpha=0.8, edgecolors='k')

        if r == 0 and c == 1:
            ax.legend(loc='upper left', fontsize=font_size)

        # optional: show per-step self-attention matrix (averaged across heads)
        if show_self_att:
            ax_self = axes[r][2]
            # attempt to recover a (dec_len, dec_len) matrix from sal
            arr = np.array(sal[i])
            mat = None
            if arr.ndim == 4:
                # (batch, heads, dec_len, dec_len)
                arr0 = arr[0]
                mat = arr0.mean(axis=0)
            elif arr.ndim == 3:
                # (batch, heads, dec_len) -> cannot form full matrix; fallback to outer product
                arr0 = arr[0]
                vec = arr0.mean(axis=0)
                mat = np.outer(vec, vec)
            elif arr.ndim == 2:
                # (heads, dec_len) -> average heads then outer
                vec = arr.mean(axis=0)
                mat = np.outer(vec, vec)
            elif arr.ndim == 1:
                mat = np.outer(arr, arr)

            if mat is not None:
                ax_self.imshow(mat, aspect='auto', origin='lower', cmap='RdBu')
                ax_self.set_title('Self-att (avg heads)')
                ax_self.set_xlabel('Decoder pos')
                ax_self.set_ylabel('Decoder pos')

        ax.set_title(f'Step {i+1}')
        ax.grid(True)

    # Hide any empty subplots
    total = nrows * ncols
    for j in range(steps, total):
        r = j // ncols
        c = j % ncols
        axes[r][c].axis('off')

    plt.tight_layout()
    if prev_fs is not None:
        plt.rcParams.update({'font.size': prev_fs})
    plt.show()


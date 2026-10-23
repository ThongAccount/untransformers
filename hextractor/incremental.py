"""Incremental (cached) decoding paths — Milestone 3.

Two arms, two different cache shapes, one contract: the incremental path must
reproduce the full forward pass **exactly** (fp32, eval mode) for every
position, because the systems comparison is only meaningful if both arms
decode the same function.

- `attention-32h`: a KV cache. State grows linearly with context:
  2 · T · d floats per layer (K and V).
- `HE` (and WE/ME): a bounded rolling state of the last `l-1` *projected*
  vectors per layer, because the offset gates only ever look back `l-1`
  positions. State is constant in context length.

The full model caps context at `l` (the positional table has `l` rows, paper
shape d=128, l=128), so the full-model decode sweep runs over 1..l. Context
lengths beyond `l` are exercised at sublayer-1 level (`bench_sublayer_scaling`),
where the comparison is about state size and per-token cost, not positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from .layers import AttentionSublayer, HESublayer, SDPAAttentionSublayer
from .model import ExtractorTransformer


# --------------------------------------------------------------------------
# per-sublayer state
# --------------------------------------------------------------------------
@dataclass
class HEState:
    """Rolling window of the last `l-1` projected vectors (HE/WE/ME)."""

    buf: Optional[torch.Tensor] = None          # [B, m, d], oldest -> newest
    m: int = 0                                  # number of vectors currently held


@dataclass
class AttentionState:
    """KV cache (attention arms)."""

    k: Optional[torch.Tensor] = None            # [B, H, T, h]
    v: Optional[torch.Tensor] = None            # [B, H, T, h]


@dataclass
class ModelCache:
    """One cache per model: sublayer-1 state per layer + position counter."""

    layers: List[object] = field(default_factory=list)
    pos: int = 0

    @classmethod
    def for_model(cls, model: ExtractorTransformer) -> "ModelCache":
        states = []
        for sub in model.sub1:
            if isinstance(sub, (AttentionSublayer, SDPAAttentionSublayer)):
                states.append(AttentionState())
            elif isinstance(sub, HESublayer):
                states.append(HEState())
            else:  # pragma: no cover — no other sublayer exists
                raise TypeError(f"unsupported sublayer {type(sub)!r}")
        return cls(layers=states)

    def state_bytes(self) -> int:
        """Bytes held by the cache (sum over layers of live tensors)."""
        total = 0
        for st in self.layers:
            if isinstance(st, AttentionState):
                for t in (st.k, st.v):
                    if t is not None:
                        total += t.numel() * t.element_size()
            elif isinstance(st, HEState):
                if st.buf is not None:
                    total += st.buf.numel() * st.buf.element_size()
        return total


# --------------------------------------------------------------------------
# sublayer-1 incremental forwards
# --------------------------------------------------------------------------
def he_forward_incremental(sub: HESublayer, X: torch.Tensor, state: HEState,
                           start: int) -> torch.Tensor:
    """HE/WE/ME sublayer over a chunk of NEW positions.

    `start` is the global position of the first row of `X` (0-based).
    The output for global position p is

        out[p] = sum_{k=0..min(p, l-1)} src[p-k] * w_ext[k]

    with `src = X @ W_in_ext` for HE and `src = X` for WE/ME — identical to
    `HESublayer.forward` restricted to those rows, because the offset window
    never reaches further back than `l-1` vectors.
    """
    B, T, _ = X.shape
    src = X @ sub.W_in_ext if sub.use_shared_projection else X
    m = state.m
    combined = torch.cat([state.buf, src], dim=1) if state.buf is not None else src

    out = torch.zeros_like(src)
    for k in range(sub.l):
        # row j of the chunk is global position p = start + j; it receives the
        # k-th offset when k <= p, and the source vector combined[:, m+j-k]
        # exists only when m+j-k >= 0.
        j_start = max(0, k - m, k - start)
        if j_start >= T:
            break
        out[:, j_start:, :] += (combined[:, m + j_start - k: m + T - k, :]
                                * sub.w_ext[k])
    if sub.use_adjustment:
        out = (X @ sub.W_adj) * out
    if sub.use_output_projection:
        out = out @ sub.W_out

    keep = min(sub.l - 1, combined.shape[1])
    if keep > 0:
        state.buf = combined[:, combined.shape[1] - keep:, :].contiguous()
        state.m = keep
    else:
        state.buf = None
        state.m = 0
    return out


def attention_forward_incremental(sub: AttentionSublayer, X: torch.Tensor,
                                  state: AttentionState, start: int) -> torch.Tensor:
    """Attention sublayer over a chunk of NEW positions (KV cache)."""
    B, T, _ = X.shape
    H, h = sub.n_heads, sub.h
    q = (X @ sub.Wq).unflatten(-1, (H, h)).transpose(-2, -3)      # [B,H,T,h]
    k = (X @ sub.Wk).unflatten(-1, (H, h)).transpose(-2, -3)
    v = (X @ sub.Wv).unflatten(-1, (H, h)).transpose(-2, -3)

    if state.k is None:
        state.k, state.v = k, v
    else:
        state.k = torch.cat([state.k, k], dim=2)
        state.v = torch.cat([state.v, v], dim=2)

    scores = q @ state.k.transpose(-1, -2) / (h ** 0.5)           # [B,H,T,Ttot]
    Ttot = state.k.shape[2]
    # cached columns (0..start-1) are all visible; the chunk is causal among
    # itself: chunk row j (global position start+j) may not see chunk columns
    # > j.
    if start < Ttot:
        mask = torch.zeros(T, Ttot, dtype=torch.bool, device=X.device)
        mask[:, start:] = torch.ones(T, Ttot - start, dtype=torch.bool,
                                     device=X.device).triu(diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    ctx = scores.softmax(dim=-1) @ state.v
    ctx = ctx.transpose(-2, -3).flatten(-2)
    return ctx @ sub.Wo


# --------------------------------------------------------------------------
# model-level incremental forward
# --------------------------------------------------------------------------
def sdpa_attention_forward_incremental(sub: SDPAAttentionSublayer, X: torch.Tensor,
                                       state: AttentionState, start: int) -> torch.Tensor:
    """Optimized (SDPA) attention sublayer over a chunk of NEW positions.

    Same function as `attention_forward_incremental` — the cache holds the same
    K/V tensors and the softmax runs over the same set of visible columns —
    but the attention *core* of a single new token is dispatched to PyTorch's
    fused `scaled_dot_product_attention` kernel.

    Why the chunk case is not fused: PyTorch's SDPA exposes an `is_causal`
    mask that is aligned to the top-left of the (query, key) matrix, and it has
    no offset-causal variant. For a chunk of new positions the visible columns
    are "all cached columns + a causal triangle inside the chunk", which the
    explicit masked formulation expresses exactly. The chunk case is the prompt
    prefill (not the timed decode step), so the fused kernel is used exactly
    where the decode latency is measured.
    """
    B, T, _ = X.shape
    H, h = sub.n_heads, sub.h
    q = (X @ sub.Wq).unflatten(-1, (H, h)).transpose(-2, -3)      # [B,H,T,h]
    k = (X @ sub.Wk).unflatten(-1, (H, h)).transpose(-2, -3)
    v = (X @ sub.Wv).unflatten(-1, (H, h)).transpose(-2, -3)

    if state.k is None:
        state.k, state.v = k, v
    else:
        state.k = torch.cat([state.k, k], dim=2)
        state.v = torch.cat([state.v, v], dim=2)

    if T == 1:
        # The single new token sees every cached column -> no mask needed.
        ctx = torch.nn.functional.scaled_dot_product_attention(
            q, state.k, state.v, is_causal=False, scale=h ** -0.5)
    else:
        scores = q @ state.k.transpose(-1, -2) / (h ** 0.5)       # [B,H,T,Ttot]
        Ttot = state.k.shape[2]
        if start < Ttot:
            mask = torch.zeros(T, Ttot, dtype=torch.bool, device=X.device)
            mask[:, start:] = torch.ones(T, Ttot - start, dtype=torch.bool,
                                         device=X.device).triu(diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
        ctx = scores.softmax(dim=-1) @ state.v
    ctx = ctx.transpose(-2, -3).flatten(-2)
    return ctx @ sub.Wo


@torch.no_grad()
def model_forward_incremental(model: ExtractorTransformer, ids: torch.Tensor,
                              cache: ModelCache) -> torch.Tensor:
    """Decode `ids` ([B, T] new positions) against `cache`; returns logits.

    Exactness contract (asserted by tests/test_m3_incremental.py): with
    dropout disabled (eval mode) the logits for a position produced here are
    identical to the logits the full forward pass produces for that position.
    """
    B, T = ids.shape
    start = cache.pos
    assert start + T <= model.l, "context exceeds l (positional table cap)"

    x = model.tok_embed[ids] * (model.d ** 0.5)
    x = x + model.pos_embed[start:start + T] * (model.d ** 0.5)
    x = model.drop(x)

    for i, (n1, s1, n2, f) in enumerate(zip(model.norms1, model.sub1,
                                            model.norms2, model.ffn)):
        n1x = n1(x)
        st = cache.layers[i]
        if isinstance(s1, SDPAAttentionSublayer):
            sub1_out = sdpa_attention_forward_incremental(s1, n1x, st, start)
        elif isinstance(s1, AttentionSublayer):
            sub1_out = attention_forward_incremental(s1, n1x, st, start)
        else:
            sub1_out = he_forward_incremental(s1, n1x, st, start)
        x_in_sub2 = n2(model.drop(sub1_out) + x)
        sub2_out = f(x_in_sub2)
        x = model.drop(sub2_out) + model.drop(sub1_out) + x

    cache.pos = start + T
    return model.head(model.final_norm(x))


@torch.no_grad()
def decode_one_token(model: ExtractorTransformer, cache: ModelCache,
                     token_id: torch.Tensor) -> torch.Tensor:
    """Single-token decode step. `token_id`: [B] int tensor."""
    return model_forward_incremental(model, token_id[:, None], cache)

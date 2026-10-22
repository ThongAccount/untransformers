"""hextractor — HE Extractor replication (arXiv:2308.07661v2, Milestone 1).

Implements the paper's causal, softmax-free Extractor sublayer family
(HE / WE / ME / SHE) and the multi-head attention baseline as drop-in
sublayer-1 replacements, under docs/milestone-0/{HE_SPEC,REPRODUCIBILITY,
TEST_SPEC}.md.

Correctness contract (HE_SPEC §8): the double-loop oracle in oracle.py is
the only accepted reference. Any vectorized implementation must match it to
<= 1e-5 max abs diff (fp32) before any fused shortcut may be introduced —
this module provides only the vectorized form, no fused kernels.
"""

from .oracle import he_oracle, attention_oracle
from .layers import HESublayer, AttentionSublayer
from .model import ExtractorTransformer
from .init import apply_init_policy
from .ops_count import count_ops_fwd, table7_counts

__all__ = [
    "he_oracle", "attention_oracle",
    "HESublayer", "AttentionSublayer",
    "ExtractorTransformer", "apply_init_policy",
    "count_ops_fwd", "table7_counts",
]

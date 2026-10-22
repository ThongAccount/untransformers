"""T13 — Op-count counter vs the paper's Table 7.

HE_SPEC §9: our convention reproduces the paper's HE/WE rows exactly; the
attention rows do not reconcile under any obvious mapping — the delta is
REPORTED (recorded in the test log), not forced.
"""

from hextractor.ops_count import table7_counts


def test_he_we_exact_match():
    c = table7_counts()
    assert c["HE_mults"] == c["paper_HE_mults"] == 7364608
    assert c["HE_total"] == c["paper_HE_total"] == 14647296
    assert c["WE_mults"] == c["paper_WE_mults"] == 5267456
    assert c["WE_total"] == c["paper_WE_total"] == 10469376


def test_attention_delta_recorded_not_forced():
    c = table7_counts()
    ours, paper = c["attention32_total_ours"], c["paper_attention32_total"]
    # Documented [U]: no obvious mapping reproduces the paper's attention rows.
    print(f"\n[attention op-count delta] ours={ours} paper={paper} "
          f"delta={paper - ours} ({paper / ours:.3f}x)")
    assert ours > 0 and paper > 0                          # both computed; equality NOT required

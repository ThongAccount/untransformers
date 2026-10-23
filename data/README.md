# `data/` — pinned corpus snapshot

Everything in this directory is a **pinned snapshot**, not a live download. The
manifest is the corpus: the shelf listing drifts over time (Project Gutenberg
is volunteer-curated), so the files and their SHA-256 checksums — not the live
shelf — define what the experiments ran on.

## Contents

| File | What it is |
|---|---|
| `manifest.json` | The corpus manifest: shelf provenance + listing hash, selection rule, per-book Gutenberg ID / URL / byte counts / SHA-256 (raw and cleaned), tokenizer record, token-array record, corpus identity, environment record. |
| `shelf20_listing.json` | Snapshot of the *shelf membership* at fetch time: all ebook IDs of bookshelf 20 in listing order + the SHA-256 of the concatenated listing pages. Proves which pool the selection was made from. |
| `corpus/*.txt` | The 100 selected books, UTF-8 plain text, PG header/footer boilerplate removed. |
| `tokenizer.json` | The trained BPE tokenizer (HF `tokenizers`), vocab 5000. |
| `tokens.npy` | The frozen token array (`uint16`) — the corpus as token ids, one EOS separator after each book. |

## Selection rule (fixed, reproducible)

1. Pool: Project Gutenberg **"Children's Literature" bookshelf 20**
   (<https://www.gutenberg.org/ebooks/bookshelf/20>) — the same source and genre
   as the paper, hand-curated and enumerable.
2. Sort the shelf's ebook IDs **ascending by numeric ebook ID**.
3. Take the **first 100**.
4. Fetch each book's plain text, strip the standard
   `*** START OF THE PROJECT GUTENBERG EBOOK ... ***` /
   `*** END OF THE PROJECT GUTENBERG EBOOK ... ***` blocks, normalize newlines.

The paper's own corpus ("top 100 books on English children's literature") has
no published list, ranking metric or release, so it cannot be reconstructed;
this book set is logged as deviation **D-01** (user-approved).

## Rebuilding / verifying

```bash
python tools/fetch_corpus.py --out data/corpus --manifest data/manifest.json --limit 100
python tools/build_tokenizer.py --corpus data/corpus --manifest data/manifest.json
python tools/fetch_corpus.py --verify-only          # re-hash files vs manifest
```

`tools/fetch_corpus.py` re-downloads from gutenberg.org; because the shelf
drifts, a rebuild in the future may select a *different* book set. That is
precisely why the checksums are committed: if a re-fetch does not reproduce
these hashes, the corpus changed and the manifest must be regenerated together
with the tokenizer and the token array.

The tokenizer and token array are deterministic functions of (corpus bytes,
library version, trainer config, seed), so `build_tokenizer.py` reproduces
`tokenizer.json` and `tokens.npy` byte-for-byte — the manifest records the
SHA-256 of both, and T14 asserts it.

## Licensing / attribution

The texts are public-domain works distributed by Project Gutenberg
(<https://www.gutenberg.org/policy/license.html>); the PG header/footer blocks
are removed here, and no Project Gutenberg trademark or endorsement is implied
by their presence in this repository. The files are redistributed unmodified
apart from that boilerplate stripping and newline normalization, so that the
experiments are reproducible offline.

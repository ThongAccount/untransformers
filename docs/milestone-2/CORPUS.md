# Milestone 2 — corpus snapshot record (T14)

Date: 2026-10-23 · Branch: `research/milestone-1-extractor` · Suite: T14
(gate-0 exit criteria 1–3 of REPRODUCIBILITY.md §8)

## What was decided

The corpus call is **closed** (user → Chief → this record): the paper's own
corpus is unreconstructable, so we use a pinned substitute from the same
source and the same genre. This document records exactly what was fetched, so
the numbers downstream can be traced to bytes.

## Provenance

| Field | Value |
|---|---|
| Source | Project Gutenberg (`https://www.gutenberg.org`) |
| Pool | "Children's Literature" bookshelf 20 |
| Pool size at snapshot | 317 books |
| Listing SHA-256 | `7fa705420b31660fd41240b58b46562c564364f2a097f7bd5a16d3c9dfc34591` (concatenated listing pages) |
| Selection rule | all shelf ebook IDs, **sorted ascending by numeric ebook ID**, first **100** |
| Books selected | 100 (ebook IDs 11 … 3796) |
| Total clean text | 32,900,247 bytes (≈ 32.9 MB) |
| Books with PG boilerplate stripped | 100 / 100 |

Per-book Gutenberg ID, source URL, raw and cleaned byte counts, and SHA-256 of
both the raw download and the cleaned file are in `data/manifest.json`; the
full shelf membership at snapshot time is in `data/shelf20_listing.json`.

## Tokenizer (REPRODUCIBILITY.md §4)

| Field | Value |
|---|---|
| Library | HF `tokenizers` 0.22.2 (version recorded in the manifest) |
| Model | BPE, `vocab_size = 5000` (paper-pinned) |
| Normalizer | none |
| Pre-tokenizer | `ByteLevel(add_prefix_space=False, use_regex=True)` (deviation D-04) |
| Decoder | `ByteLevel` |
| Special tokens | `<|endoftext|>` only |
| Trainer | `BpeTrainer(min_frequency=2, initial_alphabet=ByteLevel.alphabet())` |
| Seed | 0 (recorded; the trainer is deterministic given data order + version) |
| SHA-256 | `4e6abaa775edf70839cc6b0b881b7846e3e558c577c474fcaf22516426f6c647` |

## Token stream

| Field | Value |
|---|---|
| Token array | `data/tokens.npy`, `uint16`, one EOS after each book |
| Token count | **9,514,186** |
| SHA-256 | `fbed5914a6a8b671d23c9552b6a0d0d99996484355fc7db815606dbdcdfec8bb` |
| Corpus identity (`corpus_id`) | `fd567893d71494fdf5f733c13b4d4ac5e9ecf81c886ab298e1003d4103a5e2d7` |

The paper reports an 8.4M-token corpus **[P]**; ours is 9.51M — a 13% larger
corpus with the same source, genre and format. Logged as **D-19**. Training
budgets are compared in *tokens processed* (REPRODUCIBILITY.md §6), not epochs.

## Determinism contract (T14)

- Batches come from a `numpy.random.Generator` seeded by
  `SHA-256(train_seed | split | corpus_id | tokenizer_sha256)` over the frozen
  token array — the seed material binds the data-pipeline seed, the split, the
  corpus identity *and* the tokenizer identity, so changing any of them changes
  the stream.
- T14 asserts: identical SHA-256 of the first **100** batches' token tensors
  across two separate processes with the same seed; a different seed yields a
  different stream; the held-out stream (last 5% of the token stream) is
  disjoint from the training region at position level; the tokenizer round-trip
  is a fixed point; every committed corpus file matches its manifest checksum;
  and mutating a corpus file *reddens* the checksum check (negative control).

## Deviations introduced here

| ID | Item | Disposition |
|---|---|---|
| D-01 | Corpus book set (paper's "top 100" is unpublished) | **user-approved**: bookshelf-20, first 100 by ebook ID, SHA-256 manifest committed |
| D-19 | Corpus token count 9,514,186 vs the paper's 8.4M | accepted (book-set difference; budgets compared in tokens) |
| D-20 | `<|endoftext|>` appended after each book | accepted (document boundary; prevents windows straddling books silently) |
| D-21 | Corpus text files and the token array committed to the repo | accepted (offline verifiability of T14; public-domain texts) |

## Gate status

- [x] Corpus manifest committed (book list + SHA-256 per book + total token count).
- [x] Corpus decision recorded (user sign-off obtained).
- [x] Tokenizer JSON + SHA committed; T14 passes (no training yet).
- [x] Deterministic batch stream; identical first-100-batch SHA across processes.
- [x] Deviation ledger carries D-01/D-19/D-20/D-21.
- [x] Still **no** training run for results and **no** result interpreted at
      the time of this record.

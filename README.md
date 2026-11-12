# lisp-embed — pure Common Lisp embedding inference, IVF search, MCP server

No PyTorch. No Python. No dependencies beyond SBCL. This is a complete
embedding stack in ~4,000 lines of Common Lisp: safetensors loading, four
model families, tokenizers, a parallel embedder, an IVF index, a search
runtime, and an MCP server for agents.

## What's in here

| File | Role |
|---|---|
| `json.lisp` | JSON parser/writer (own implementation) |
| `st.lisp` | safetensors reader — F32, BF16, F16 |
| `tokenizer.lisp` | WordPiece, byte-level BPE (GPT-2 and Llama-3 pre-split), Unigram; `tokenizer.json` loader with post-processor specials |
| `bert.lisp` | BERT / RoBERTa / DistilBERT / MPNet (relative-position attention), CLS/mean pooling, query prefixes |
| `llama.lisp` | decoder-only embedding models: RMSNorm, RoPE, GQA, SwiGLU, causal attention, last-token pooling (Qwen3-Embedding, Llama, Mistral, Gemma) |
| `ann.lisp` | IVF index: k-means + inverted lists, pure Lisp |
| `embed.lisp` | parallel torch-free embedder for a chunk corpus |
| `extract.lisp` / `chunk.lisp` | corpus extraction and paragraph-aligned chunking |
| `kb.lisp` | search runtime: flat vector store, IVF query, injection blocks |
| `serve.lisp` | MCP stdio server (`kb_search`, `kb_inject`, `kb_stats`) |
| `config.lisp` | paths and model selection via environment |
| `bench/` | benchmark harness and results |

## Model support

`bert:load-model` dispatches on `model_type` from `config.json`:

| Family | model_type | Architecture | Pooling |
|---|---|---|---|
| BERT | bert, distilbert | LayerNorm, absolute positions | CLS / mean |
| RoBERTa | roberta | as BERT, position offset 2, `<s>`/`</s>` | CLS / mean |
| MPNet | mpnet | relative-position attention (T5 buckets), post-LN residuals | mean |
| Llama | qwen3, qwen2, llama, mistral, gemma | RMSNorm, RoPE, GQA, SwiGLU, causal | last token |

Verified against the reference implementations (transformers): bge-small
golden `-0.08250509, -0.050689265, ...`; all-mpnet-base-v2 cosines
0.7655 / -0.0505; Qwen3-Embedding-0.6B 0.773 / 0.3661 — all reproduced exactly.

## Benchmark — cybersec retrieval (13 models)

Method: 10 domain queries, candidate pool = the live index's top-20 per query;
each model re-ranks the top-10; relevance = keyword labels; P@5 = mean fraction
of the top-5 that are on-topic; query latency = single embed, min of 3.

| Model | Dim | P@5 | top-1 | Query ms | Index GB | Rebuild |
|---|---|---|---|---|---|---|
| e5-base-v2 | 768 | **0.920** | 1.000 | 771 | 10.1 | 4x |
| gte-base | 768 | 0.880 | 1.000 | 672 | 10.1 | 4x |
| mxbai-embed-large-v1 | 1024 | 0.880 | 1.000 | 4001 | 13.4 | 14x |
| all-MiniLM-L6-v2 | 384 | 0.860 | 0.800 | **90** | 5.0 | **0.5x** |
| all-mpnet-base-v2 | 768 | 0.860 | 0.900 | 600 | 10.1 | 4x |
| gte-small | 384 | 0.840 | 0.900 | 170 | 5.0 | 1x |
| snowflake-arctic-embed-s | 384 | 0.840 | 0.900 | 159 | 5.0 | 1x |
| bge-small-en-v1.5 | 384 | 0.820 | 0.800 | 157 | 5.0 | 1x |
| e5-small-v2 | 384 | 0.820 | 1.000 | 198 | 5.0 | 1x |
| bge-base-en-v1.5 | 768 | 0.800 | 1.000 | 614 | 10.1 | 4x |
| bge-large-en-v1.5 | 1024 | 0.820 | 0.900 | 2245 | 13.4 | 14x |
| Qwen3-Embedding-0.6B | 1024 | 0.820 | 0.800 | 2933 | 13.4 | 17x |
| snowflake-arctic-embed-m | 768 | 0.660 | 0.500 | 601 | 10.1 | 4x |

Index GB = 3.27M rows x dim x 4 bytes. Rebuild = relative FLOPs vs a
12-layer 384d model (the full corpus takes CPU-days at 8 threads).

Takeaways: bigger is not better — the 1024d tier gains nothing over 768d and
costs 3-4x per query; e5-base-v2 leads quality; MiniLM-L6 is the best
quality-per-cost (6 layers, 90ms queries, half the rebuild of a 12-layer model).

## Performance: optional BLAS fast path

The forward pass is pure Lisp by default. Set `KB_BLAS` to an OpenBLAS DLL
(e.g. the one shipped with SciPy) and `bert:load-blas` switches the linear
layers and attention to `cblas_sgemm`/`sgemv`:

| Path | 256-char chunk | 512-char chunk |
|---|---|---|
| pure Lisp | ~1.6 s | ~8.1 s |
| BLAS (1 thread) | ~0.34 s | ~0.70 s |
| BLAS + 16 SBCL threads | 54-64 chunks/s | — |

A 2.79M-chunk corpus rebuild drops from ~16 days to ~14 hours. The pure path
stays the default and produces identical vectors (verified: `0.02233232` vs
`0.022332346`).

## Quick start

```lisp
;; build
(sbcl --load build.lisp)

;; embed
(load "config.fasl") (load "json.fasl") (load "st.fasl")
(load "tokenizer.fasl") (load "llama.fasl") (load "bert.fasl")
(bert:load-model (cfg:model-path))
(coerce (subseq (bert:embed "harden the SSH server") 0 5) 'list)
```

Environment: `KB_DATA_DIR` (default `./data/`), `KB_MODEL` (defaults to
all-MiniLM-L6-v2 in the HuggingFace cache), `KB_SRC` (corpus root).

Pipeline: `extract.lisp` -> `chunk.lisp` -> `embed.lisp` -> `ann.lisp`,
then `kb.lisp` for search and `serve.lisp` for the MCP server.

## Benchmark harness

`bench/test-pool.lisp` extracts a candidate pool from a live index;
`bench/test-bench.lisp` re-ranks it with each model and scores P@5.
`bench/bench.out`, `bench/bench2.out` are the raw runs behind the table.

## License

MIT.

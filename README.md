# vllm-spark-nvfp4

Official vLLM + b12x kernels + **NVFP4 MLA KV cache** for DeepSeek-V4 on
2x DGX Spark (GB10, `sm_121a`).

Everything here starts from a **pinned official release image**. The only
non-upstream content is one public PyPI wheel and one patch, both documented
below and in the Dockerfile header.

```bash
docker build -t vllm-spark-nvfp4:v0.27.1-b12x .
```

---

## Why this exists

`--kv-cache-dtype nvfp4_ds_mla` roughly halves KV bytes per token versus
`fp8_ds_mla` (a 584-byte padded page instead of 576-byte fp8 pages holding far
fewer tokens per GiB), which is the difference between a ~1.7M and a ~2.5M token
KV pool on a 2x GB10 box. It is **not in upstream vLLM**, and no upstream PR
exists for it.

The implementations that do have it are third-party images built from unmerged
branches. This repo reproduces the capability on top of an official release
instead, so the lineage is auditable.

## What goes into the image

| # | Component | Source | Status |
|---|---|---|---|
| 1 | `vllm/vllm-openai:v0.27.1` | official release image | pinned |
| 2 | `b12x==1.2.6` | public PyPI, by Luke Alonso | *"SM120-only CuTe DSL kernels for NVFP4 GEMM and MoE"* |
| 3 | NVFP4 MLA KV patch | **ours**, 191 lines, 7 files | no upstream PR exists |
| 4 | b12x dense linear | vllm-project/vllm **PR #52016** | **merged** upstream Aug 14 |
| 5 | b12x FP4/MXFP4 MoE | vllm-project/vllm **PR #52018** | **open**, `local-inference-lab:dev/b12x-moe` |

Items 3-5 ship as one verified diff, `combined-v0.27.1.patch`
(29 files, 3038 lines, applies to pristine v0.27.1 with exit 0). The individual
inputs are kept under `patches/` for review.

### Why the b12x PRs are needed

DeepSeek-V4's experts are **MXFP4**, and stock v0.27.1 has no b12x entry in its
MXFP4 MoE backend list (`deep_gemm, flashinfer_trtllm, flashinfer_cutlass,
triton, humming, marlin, aiter...`). Without them:

```
ValueError: moe_backend='b12x' is not supported for MXFP4 MoE
```

Upstream wires b12x MXFP4 through `fused_moe/oracle/mxfp4.py`, exposing
`B12X_MXFP4_MXFP8` and `B12X_MXFP4_BF16`. PR #52018 depends on infrastructure
from #52016, so both are cherry-picked.

### Deliberately dropped hunks

4 of #52018's 13 hunks are omitted because they target code that does not exist
in 0.27.1, and none affect MoE dispatch:

- `oracle/nvfp4.py`: `gemm1_alpha` / `gemm1_beta` swiglu args absent from
  0.27.1's `nvfp4_w4a16_moe_quant_config` signature
- `warmup/b12x_warmup.py`: warmup token-count tuning and a log-string change

The shipped combined patch contains only what applied and verified, so the build
has no rejects.

## The NVFP4 KV patch

Pure Python, no CUDA compilation. It adds a **584-byte padded uint8 envelope**
next to the existing 576-byte `fp8_ds_mla` one, reusing the same paged
machinery:

```python
if kv_cache_dtype in ("nvfp4", "nvfp4_ds_mla"):
    cache_config.cache_dtype = "nvfp4_ds_mla"
    return "nvfp4_ds_mla", torch.uint8
...
alignment = _dsv4_page_alignment(self.kv_cache_dtype)  # 584 / 576 / 512
```

v0.27.1 already ships the surrounding NVFP4 plumbing (`KVQuantMode`,
`nvfp4_kv_cache_full_dim`, `nvfp4_split_data_scale`, the ModelOpt dtype map) and
has already narrowed its nvfp4+MLA guard to `cache_dtype == "nvfp4"`, so the
patch only adds the DeepSeek-V4 variant.

Derived from two independent out-of-tree implementations that agree on the
584-byte envelope: `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (vLLM 0.25.2) and a
locally built `dspark-nvfp4-stage-c` (vLLM 0.21.1rc1). anemll was the donor: it
routes the dtype through the `KVQuantMode` enum rather than hardcoding
`head_bytes`, and covers both the `flashmla_sparse` and `sparse_swa` backends.

> On older bases (0.26.1, and forks built from pre-0.27 branches) the guard is
> still `cache_dtype.startswith("nvfp4")`, which wrongly rejects
> `nvfp4_ds_mla`. Those need an additional one-line fix.

## Build-time assertions

`patch` exiting 0 does not prove the dtype is reachable, and a silently
unreachable dtype is the exact failure this image exists to avoid. The build
asserts behaviour:

```python
import b12x
assert "nvfp4_ds_mla" in CacheDType.__args__
assert _dsv4_page_alignment("nvfp4_ds_mla") == 584
assert _dsv4_page_alignment("fp8_ds_mla")   == 576   # no regression
```

A vLLM bump that moves this code fails the build rather than shipping an image
that ignores the flag.

## Usage

```bash
vllm serve <deepseek-v4-model> \
    --tensor-parallel-size 2 --nnodes 2 --node-rank {0,1} \
    --kv-cache-dtype nvfp4_ds_mla \
    --moe-backend b12x --linear-backend b12x \
    --block-size 256 --max-model-len 1048576 \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'
```

Confirm in the log, rather than assuming:

```
Using DeepSeek V4 padded nvfp4_ds_mla KV cache format
GPU KV cache size: N tokens
```

Note `--moe-backend b12x` (the #52018 FP4 MoE kernels), not `flashinfer_b12x`
(the older NVFP4-only CuteDSL path).

## Not included

- **eugr / spark-arena images.** Those build from unmerged b12x dev branches,
  not from a release or from main. Their `0.1.devNNNNN` version string is a
  tagless commit count, not a newer vLLM.
- **SSD KV offload.** `OffloadingConnector` with an `fs_python` disk tier dies
  with a CUDA illegal memory access on this model, with and without b12x
  attention. Not worked around: that class of bug can corrupt output silently
  rather than crash.

## Upstreaming

The b12x half is already heading upstream (#52016 merged, #52018 open), so those
layers become unnecessary as they land. The NVFP4 MLA KV half has no upstream
home yet and is the part worth proposing.

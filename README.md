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
| 6 | mHC DeepGEMM guard | vllm-project/vllm **PR #50645** | **open**; without it the worker dies on sm121, see [MHC_DEEPGEMM_SM121.md](MHC_DEEPGEMM_SM121.md) |
| 7 | KV-offload bounds check | vllm-project/vllm **PR #53271** (ours) | **open**; diagnosability only, see [KV_OFFLOAD_MLA.md](KV_OFFLOAD_MLA.md) |

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

### Per-site dtype binding

`models/deepseek_v4/attention.py` computes the page alignment in two different
classes, each deriving the KV dtype from a different place:

| site | class | dtype source |
|---|---|---|
| ~663 | `DeepseekV4Attention` | `self.kv_cache_dtype` |
| ~698 | `DeepseekV4IndexerCache` | `vllm_config.cache_config.cache_dtype` |

The patch binds each site's own expression to a local `_dsv4_cache_dtype` and
passes that to the alignment helper. Substituting `self.kv_cache_dtype` at both
sites looks correct and builds fine, but fails at runtime with:

```
'DeepseekV4IndexerCache' object has no attribute 'kv_cache_dtype'
```

### Deliberately dropped hunks

4 of #52018's 13 hunks are omitted because they target code that does not exist
in 0.27.1, and none affect MoE dispatch:

- `oracle/nvfp4.py`: `gemm1_alpha` / `gemm1_beta` swiglu args absent from
  0.27.1's `nvfp4_w4a16_moe_quant_config` signature
- `warmup/b12x_warmup.py`: warmup token-count tuning and a log-string change

The shipped combined patch contains only what applied and verified, so the build
has no rejects.

**One thing #52018 references but does not define:** `B12xWarmupUnit`, imported
by `fused_moe/b12x.py` from `vllm.utils.b12x`. It landed on upstream main in a
separate commit that is in neither PR nor in v0.27.1, so without it the image
builds cleanly and then dies at import:

```
cannot import name 'B12xWarmupUnit' from 'vllm.utils.b12x'
```

The patch adds the dataclass verbatim from main.

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

### Backend naming: `b12x` vs `flashinfer_b12x`

Both names exist, both are accepted by `--moe-backend`, and picking the wrong
one produces an error that reads like the patch failed:

```
ValueError: moe_backend='flashinfer_b12x' is not supported for MXFP4 MoE.
Expected one of ['deep_gemm', 'flashinfer_trtllm', ..., 'marlin', ...]
```

That message lists the *stock* MXFP4 backends and says nothing about b12x, so it
looks like the cherry-pick did not land. It did. From `config/kernel.py`:

| value | meaning |
|---|---|
| **`b12x`** | native b12x FP4 MoE kernels on SM12x, added by **#52018**. **Use this.** |
| `flashinfer_b12x` | FlashInfer CuteDSL fused MoE, the older **NVFP4-only** path |

DeepSeek-V4's experts are MXFP4, so only `b12x` reaches them. The MXFP4 backends
#52018 registers are `B12X_MXFP4_MXFP8` and `B12X_MXFP4_BF16`, dispatched via
`B12X_BACKENDS` in `fused_moe/oracle/mxfp4.py`.

To check the cherry-pick landed, independently of any flag:

```bash
docker run --rm --entrypoint bash <image> -lc \
  'grep -c B12X_BACKENDS /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py'
```

Same applies to `--linear-backend b12x` (native B12X FP8/FP4 linear kernels,
from #52016) versus `flashinfer_b12x` (FlashInfer b12x CuteDSL NVFP4 GEMM).

## Settings the fork images bake in, and stock ones do not

Moving to official lineage means inheriting upstream defaults instead of a
fork's GB10 tuning. Two that surface immediately:

| Symptom | Cause | Fix |
|---|---|---|
| `Assertion error (deepgemm-src/csrc/apis/layout.hpp:60): Unknown SF transformation` | DeepGEMM linear path gets a scale layout it does not recognise | `--linear-backend b12x` to route around DeepGEMM, and/or `VLLM_USE_DEEP_GEMM_E8M0=1` |
| `moe_backend=... is not supported for MXFP4 MoE` | wrong backend name, see above | `--moe-backend b12x` |

`VLLM_USE_DEEP_GEMM_E8M0=1` is described in bjk110's DeepSeek-V4 preset as
mandatory for the SM121 numerical contract. Note that on its own it did **not**
clear the assertion here; the linear backend mattered.

## Not included

- **eugr / spark-arena images.** Those build from unmerged b12x dev branches,
  not from a release or from main. Their `0.1.devNNNNN` version string is a
  tagless commit count, not a newer vLLM.
- **SSD KV offload.** `OffloadingConnector` with an `fs_python` disk tier dies
  with a CUDA illegal memory access on this model, with and without b12x
  attention. Not worked around: that class of bug can corrupt output silently
  rather than crash. Root cause and the upstream fix are documented in
  [KV_OFFLOAD_MLA.md](KV_OFFLOAD_MLA.md); the diagnosability half is submitted as
  [vllm-project/vllm#53271](https://github.com/vllm-project/vllm/pull/53271).

## Upstreaming

The b12x half is already heading upstream (#52016 merged, #52018 open), so those
layers become unnecessary as they land. The NVFP4 MLA KV half has no upstream
home yet and is the part worth proposing.

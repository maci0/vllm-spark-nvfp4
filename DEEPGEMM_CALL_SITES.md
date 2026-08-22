# DeepGEMM call sites that break DeepSeek-V4 on SM120-family GPUs

DeepGEMM has no SM120-family kernels. Upstream vLLM guards its DeepSeek-V4
DeepGEMM calls **inconsistently**: some are unguarded, some use a predicate that
does not check the architecture. On GB10 each unguarded site aborts the worker
with

```
Assertion error (/workspace/.deps/deepgemm-src/csrc/apis/<file>.hpp:<n>):
Unsupported architecture
```

They surface one at a time: fix one, boot further, hit the next.

---

## The sites, in the order they are reached

| # | Site | File | Guard upstream | Fix |
|---|---|---|---|---|
| 1 | mHC pre-broadcast | `kernels/mhc/tilelang.py` | **none** | cherry-pick [#50645](https://github.com/vllm-project/vllm/pull/50645) |
| 2 | Output projection | `models/deepseek_v4/nvidia/flashinfer_sparse.py` | **none** | ours: b12x WO, else BF16 reference |
| 3 | Sparse-MLA indexer | `v1/attention/backends/mla/indexer.py` | **wrong predicate** | ours: `has_deep_gemm()` -> `is_deep_gemm_supported()` |

`deep_gemm_mega_moe` paths in `models/deepseek_v4/nvidia/model.py` also call
DeepGEMM, but they are gated by MoE backend selection and are not reached when
running `--moe-backend b12x`.

### The predicate distinction

```python
def has_deep_gemm() -> bool:            # module importable?
def is_deep_gemm_supported() -> bool:   # importable AND env-enabled AND arch supported
```

Site 3 used `has_deep_gemm()`, which is true on GB10 (the module imports fine),
so the guard passed and the call aborted anyway. Upstream PR
[#53055](https://github.com/vllm-project/vllm/pull/53055) is titled "enforce
compute capability check", which is the same defect class.

Note also that `VLLM_USE_DEEP_GEMM=0` does **not** avoid unguarded sites: it is
only consulted inside `is_deep_gemm_supported()`.

## Site 2: output projection

`_o_proj` is the hot one. It runs on **every layer of every forward**, unlike the
mHC pre-broadcast which runs once on the first layer, so the fallback choice
matters for throughput rather than just for booting.

Three tiers, in preference order:

```python
if not is_deep_gemm_supported():
    out = _b12x_wo_projection(self, o, positions)   # 1. b12x fused kernel
    if out is not None:
        return out
    return _o_proj_reference(self, o, positions)    # 2. BF16 reference
return deep_gemm_fp8_o_proj(...)                    # 3. upstream, unchanged
```

**Tier 1, b12x.** `b12x.gemm.wo_projection` fuses inverse-RoPE with the WO-A and
WO-B GEMMs. It has a direct-call mode (`run_inv_rope`) taking raw tensors plus
packed weights, so the full `plan`/`bind` scratch lifecycle is not required, and
the packed weights carry both WO-A and WO-B, so the whole projection is one call.
Weights are packed once per layer and cached.

Dimensions passed to `pack_weights`:

```
groups       = layer.n_local_groups
group_width  = (n_local_heads // n_local_groups) * head_dim
rank         = layer.o_lora_rank
hidden       = layer.hidden_size
```

`group_width` was derived from the XPU path's einsum shapes
(`torch.einsum("tgd,grd->tgr")` over `o.view(tokens, n_local_groups, -1)`).
**Confirmed correct in practice:** the profiling forward pass completed with a
1,597,380-token KV pool allocated and zero fallback warnings logged.

**Tier 2, BF16 reference.** Delegates to `rocm_inv_rope_einsum`, vLLM's existing
reference for this computation, which imports cleanly on CUDA because it is plain
torch. Reused rather than reimplemented: hand-writing inverse-RoPE and the WO-A
scale handling is where a subtle error produces wrong tokens instead of a crash.

Any exception in tier 1 disables b12x for that layer, logs once, and degrades to
tier 2. A kernel path should not be able to kill the worker, and a wrong answer
here would be silent.

## Why not port eugr's integration

eugr's fork carries 8 `VLLM_USE_B12X_*` switches threaded through `envs.py`,
`deepseek_v4/attention.py` (+408 lines), `deepseek_v4/nvidia/model.py` (+537),
and a `compilation/b12x_capture.py` that pins b12x kernel resolution for CUDA
graph capture. It works, but it is ~1000 lines from an unmerged branch with no
upstream PR.

The fixes here total well under 200 lines, use only in-tree code plus b12x's
public API, and are shaped like the PRs upstream is already merging for this
class of bug, so they are individually upstreamable.

## Known remaining risk

The b12x direct-call mode allocates internally, so it may not be CUDA-graph
capture safe. That is what eugr's `bind_inv_rope` + `b12x_capture.py` exist to
solve. If capture fails, the fix is to move tier 1 onto the bind lifecycle
rather than to abandon the fast path.

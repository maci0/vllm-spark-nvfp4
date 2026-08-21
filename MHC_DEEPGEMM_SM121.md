# DeepSeek-V4 mHC needs DeepGEMM kernels the official image does not ship

On GB10 (`sm_121a`), booting DeepSeek-V4 from a stock `vllm/vllm-openai` image
aborts the worker during the memory-profiling forward pass:

```
RuntimeError: Worker failed with error 'Assertion error
(/workspace/.deps/deepgemm-src/csrc/apis/hyperconnection.hpp:56):
Unsupported architecture'
```

This is **not** a configuration problem, and `VLLM_USE_DEEP_GEMM=0` does not
avoid it.

---

## The call chain

The head log only shows executor frames. The real stack is in the worker's
in-container log (`/tmp/sparkrun_serve.log`):

```
models/deepseek_v4/nvidia/model.py:914   forward
  -> kernels/mhc/tilelang.py:374         mhc_pre_broadcast_tilelang
     -> utils/deep_gemm.py:635           tf32_hc_prenorm_gemm
        -> deepgemm hyperconnection.hpp:56  Unsupported architecture
```

DeepSeek-V4's **mHC (multi-head connection) pre-broadcast** calls DeepGEMM's
`tf32_hc_prenorm_gemm` unconditionally. In v0.27.1 that call site has no
capability guard:

```python
from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm

tf32_hc_prenorm_gemm(
    residual_flat, fn_broadcast, gemm_out_mul, gemm_out_sqrsum, n_splits,
)
```

`deep_gemm` is not an importable Python package in these images: the symbols are
compiled into the extension, which is why the assertion carries a build path
(`/workspace/.deps/deepgemm-src/`). **The DeepGEMM pinned by official vLLM
builds has no SM120-family kernels for this API**, so no Python-level setting can
route around it.

## This is a known upstream issue

| Ref | Title | State |
|---|---|---|
| [#51959](https://github.com/vllm-project/vllm/issues/51959) | **"DeepGEMM pin has no SM120 kernels: family-12 Blackwell cannot run hyperconnections"** | closed 2026-08-13 |
| [#50645](https://github.com/vllm-project/vllm/pull/50645) | "[Bugfix] Guard `mhc_pre_broadcast_tilelang` on DeepGEMM support" | open |
| [#53055](https://github.com/vllm-project/vllm/pull/53055) | "guard DeepGEMM in MHC pre-broadcast, enforce compute capability check" | open |

#51959 describes exactly this hardware and this symptom. That the fork images
work is explained by their bundling a ported DeepGEMM: anemll ships
`/opt/b12x/docs/sm120_dense_fp8_deepgemm_port.md`.

## The fix this image uses

Cherry-picks **PR #50645**, which guards the call on a capability check and
falls back to the TileLang kernel:

```python
use_deep_gemm = is_deep_gemm_supported()
if use_deep_gemm:
    n_splits = compute_num_split(64, hidden_size, cdiv(num_tokens, 64))
else:
    n_splits = 1          # falls back to _tilelang_hc_prenorm_gemm
```

The same guard is applied to both call sites in `kernels/mhc/tilelang.py`.

### A bespoke fallback was written first and then discarded

Before finding #50645, this repo carried its own fallback: catch the
`RuntimeError`, set a sticky flag, and dispatch to `_torch_hc_prenorm_gemm`
(`out = x.float() @ fn.T`, `sqrsum = x.float().square().sum(-1)`).

It worked, but upstream's is better on every axis, so it was removed:

| | bespoke | **#50645** |
|---|---|---|
| detection | catch `RuntimeError` at call time | **`is_deep_gemm_supported()`** capability check |
| fallback | torch, fp32 matmul on the host path | **`_tilelang_hc_prenorm_gemm`**, a GPU kernel |
| `n_splits > 1` | raised `NotImplementedError` | **handled**, sets `n_splits = 1` |
| tests | none | ships `tests/kernels/test_mhc_kernels.py` |

Catching an exception to infer hardware capability is the weaker pattern: it
only discovers the problem by triggering it, and it risks masking unrelated
runtime errors that happen to carry the same message.

## Consequence for building on official images

DeepSeek-V4 on family-12 Blackwell needs, from an official base:

1. **#50645** (or #53055) to guard the mHC DeepGEMM call, otherwise the worker
   dies during profiling; and
2. the **b12x** kernels for MoE and linear, since the stock MXFP4 backend list
   has no b12x entry.

Both are pure Python and cherry-pickable. What is *not* reachable this way is
DeepGEMM itself: if a future workload needs the fused hyperconnection kernel
rather than the TileLang fallback, that requires a DeepGEMM build with SM120
kernels, which is a compiled dependency of the vLLM image.

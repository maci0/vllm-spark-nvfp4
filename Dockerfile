# DeepSeek-V4 on 2x DGX Spark (GB10, sm_121a):
#   official vLLM image + b12x kernels + NVFP4 MLA KV cache.
#
# ============================== PROVENANCE ==============================
#
# Everything non-upstream in this image is listed here. Nothing else is added.
#
# 1. BASE  vllm/vllm-openai:${VLLM_RELEASE}
#      Official image, pinned. Default v0.27.1 (the latest release on PyPI at
#      time of writing). Build with VLLM_RELEASE=nightly to get the b12x MoE
#      layer below, which needs infrastructure merged to main after 0.27.1.
#
# 2. WHEEL  b12x==${B12X_VERSION}   (public PyPI)
#      "Unapologetically SM120-only CuTe DSL kernels for NVFP4 GEMM and MoE",
#      by Luke Alonso. Upstream vLLM already carries integration code for it
#      (model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py); only
#      the runtime wheel is missing from the stock image.
#
# 3. PATCH  ours: nvfp4-ds-mla (191 lines, 7 files)
#      Shipped as part of combined-v0.27.1.patch together with item 4.
#      Adds kv_cache_dtype=nvfp4_ds_mla for DeepSeek-V4: a 584-byte padded
#      uint8 KV envelope alongside the existing 576-byte fp8_ds_mla one.
#      Pure Python, no CUDA compilation.
#
#      Not upstream, and no upstream PR exists for it (searched
#      vllm-project/vllm). Derived by reading two independent out-of-tree
#      implementations that agree on the 584-byte envelope:
#        - ghcr.io/anemll/dspark-vllm-gx10:0.1.1        (vLLM 0.25.2)
#        - vllm-dspark-runtime:dspark-nvfp4-stage-c     (vLLM 0.21.1rc1)
#      anemll was used as the donor: it routes the dtype through the
#      KVQuantMode enum rather than hardcoding head_bytes, and covers both the
#      flashmla_sparse and sparse_swa backends.
#
#      v0.27.1 already ships the surrounding NVFP4 plumbing (KVQuantMode,
#      nvfp4_kv_cache_full_dim, nvfp4_split_data_scale, the ModelOpt dtype map)
#      and has already narrowed its nvfp4+MLA guard to `cache_dtype == "nvfp4"`,
#      so this patch only adds the DeepSeek-V4 variant. Earlier versions
#      (0.26.1, and the eugr fork) still use `startswith("nvfp4")`, which
#      wrongly rejects nvfp4_ds_mla; on those a guard fix is also required.
#
# 4. PATCH  b12x linear + MoE, cherry-picked from two upstream PRs by
#      Luke Alonso (b12x's author), both against vllm-project:main:
#        #52016 "[Kernel] Add B12X dense linear backends"  MERGED Aug 14
#        #52018 "[Kernel] Add b12x FP4 MoE backend"        OPEN
#                head branch local-inference-lab:dev/b12x-moe
#      Both are pure Python. Filtered to vllm/** only: their tests/, docs/,
#      setup.py and CI config have no place in an installed package.
#
#      Needed because DeepSeek-V4's experts are MXFP4 and stock v0.27.1 has no
#      b12x entry in its MXFP4 MoE backend list (deep_gemm, flashinfer_trtllm,
#      flashinfer_cutlass, triton, humming, marlin, aiter...). Without it,
#      --moe-backend flashinfer_b12x fails with "not supported for MXFP4 MoE".
#      Upstream wires b12x MXFP4 through fused_moe/oracle/mxfp4.py.
#
#      #52016 applies to v0.27.1 cleanly. #52018 then lands 9 of 13 hunks; the
#      4 rejects are non-essential and were dropped deliberately:
#        - oracle/nvfp4.py: gemm1_alpha/gemm1_beta swiglu args that do not
#          exist in 0.27.1's nvfp4_w4a16_moe_quant_config signature
#        - b12x_warmup.py: warmup token-count tuning and a log-string change
#      The shipped combined patch contains only what applied and verified.
#
# NOT INCLUDED, deliberately:
#   - eugr / spark-arena images. Those build from Luke Alonso's unmerged b12x
#     dev branches, not from any release or from main. Their version string
#     (0.1.devNNNNN) is a tagless commit count, not a newer vLLM.
#   - SSD KV offload. OffloadingConnector dies with a CUDA illegal memory
#     access on this model, with and without b12x attention. Not worked around:
#     that bug class can corrupt output silently rather than crash.
#
# ============================== BUILD ==================================
#   docker build -t vllm-spark-nvfp4:v0.27.1 .
#
# combined-v0.27.1.patch = item 3 + item 4, regenerated as a single diff against
# pristine v0.27.1 so it applies in one step with no rejects:
#   29 files, 3038 lines, verified apply exit 0.
# =======================================================================

ARG VLLM_RELEASE=v0.27.1
FROM vllm/vllm-openai:${VLLM_RELEASE}

ARG B12X_VERSION=1.2.6
RUN pip install --no-cache-dir "b12x==${B12X_VERSION}"

# --- 3+4. One combined patch: NVFP4 MLA KV (ours) + b12x linear/MoE (upstream PRs) ---
# Verified to apply to pristine v0.27.1 with exit 0: 29 files, 3038 lines.
COPY combined-v0.27.1.patch /tmp/
RUN VLLM_DIR="$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')" \
 && patch -p1 --no-backup-if-mismatch -d "$(dirname "$VLLM_DIR")" < /tmp/combined-v0.27.1.patch \
 && rm -f /tmp/combined-v0.27.1.patch

# A clean `patch` exit does not prove the dtype is reachable. A silently
# unreachable dtype is exactly the failure this image exists to avoid, so assert
# behaviour rather than trusting the patch tool.
RUN python3 - <<'PY'
import b12x  # noqa: F401
from vllm.config.cache import CacheDType
opts = getattr(CacheDType, "__args__", ())
assert "nvfp4_ds_mla" in opts, f"nvfp4_ds_mla not an accepted --kv-cache-dtype: {opts}"
from vllm.models.deepseek_v4 import attention as a
assert a._dsv4_page_alignment("nvfp4_ds_mla") == 584, "wrong NVFP4 page envelope"
assert a._dsv4_page_alignment("fp8_ds_mla") == 576, "fp8_ds_mla envelope regressed"
print("image OK: b12x importable, nvfp4_ds_mla accepted, 584B envelope wired")
PY

LABEL org.opencontainers.image.title="vllm-spark-nvfp4"
LABEL org.opencontainers.image.description="Official vLLM + b12x kernels + nvfp4_ds_mla KV for DeepSeek-V4 on GB10 (sm121)"
LABEL dev.spark.nvfp4-patch="nvfp4-ds-mla, 584B envelope, ours, not upstream"
LABEL dev.spark.b12x-patch="vllm-project/vllm PR #52016 (merged) + #52018 (open), cherry-picked onto v0.27.1"

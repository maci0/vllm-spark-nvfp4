# NVFP4 KV for DeepSeek-V4 on the eugr b12x image

An 89-line patch across 4 files that makes `nvfp4_ds_mla` work for DeepSeek-V4
on `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-b12x`.

**Status: works, not competitive.** It allocates, passes the forward pass, and
costs 22% fewer bytes per token than `fp8_ds_mla`. It has never served a
request, because CUDA graph capture then runs out of memory. See
[§4](#4-why-it-is-not-a-better-production-config) before adopting it.

Build: [`Dockerfile.eugr-nvfp4`](Dockerfile.eugr-nvfp4) ·
Patch: [`eugr-nvfp4.patch`](eugr-nvfp4.patch)

---

## 1. Why it is small

eugr's tree already carries `nvfp4_ds_mla` through the generic MLA stack
(`mla.py`, `mla_cache_format.py`, `mla_attention.py`, `b12x_mla_sparse.py`,
`_custom_ops.py`) and already classifies it as `KVQuantMode.NVFP4`. That last
point matters: on stock vLLM 0.27.1 the same dtype returns `KVQuantMode.NONE`,
which silently disables the packed layout. eugr had already fixed it.

Only the `deepseek_v4` module and the two DSv4 page-size specs hardcode fp8.

## 2. The four changes

| # | file | change |
|---|---|---|
| 1 | `config/vllm.py` | the nvfp4-with-MLA guard used `startswith("nvfp4")`, which also caught `nvfp4_ds_mla`; narrowed to exact `"nvfp4"` |
| 2 | `models/deepseek_v4/attention.py` | the DSv4 dtype resolver asserts fp8 before it can return the padded layout; `nvfp4_ds_mla` now resolves to `("nvfp4_ds_mla", torch.uint8)` |
| 3 | `models/deepseek_v4/attention.py`, `v1/attention/backends/mla/sparse_swa.py` | page alignment: `nvfp4_ds_mla` joins the `fp8_ds_mla` predicate |
| 4 | `v1/kv_cache_interface.py` | the two DSv4 **584B page-size branches** tested `fp8_ds_mla` exactly |
| 5 | `v1/attention/backends/mla/sparse_swa.py` | `get_kv_cache_shape` must recognise `nvfp4_ds_mla` |

**The governing rule, and the one that took five runs to find:**
`nvfp4_ds_mla` is the *same* 584-byte paged envelope as `fp8_ds_mla`, differing
only in how each token's bytes are encoded. Every layout decision therefore
follows the fp8 path. Two tempting deviations are both wrong:

- **Giving it its own `alignment=584`.** 584 is not a multiple of 16. It leaves
  the page unpadded and puts sub-tensor pointers on 8-byte boundaries.
- **Letting it fall into the generic `KVQuantMode.NVFP4` page-size branch.**
  That models a per-head packed layout (`head//2` data + `head//16` scales =
  288B for `head_size` 512), which is not this envelope.

Either produces a page size that disagrees with the tensor shape, and the
mismatch surfaces as

```
RuntimeError: Tensor data pointer is not aligned to 16 bytes.
```

raised deep in the forward pass, long after the cache has allocated cleanly.
Concretely: page-size math said 32,832 bytes while `get_kv_cache_shape` said
37,440.

## 3. Build-time assertions

Every failure mode above costs a full two-node model load (~8 minutes) before
it appears, so [`Dockerfile.eugr-nvfp4`](Dockerfile.eugr-nvfp4) asserts
behaviour instead of trusting `patch` to exit 0:

```python
assert a._resolve_dsv4_kv_cache_dtype(True, "nvfp4_ds_mla", None) == ("nvfp4_ds_mla", torch.uint8)
assert a._resolve_dsv4_kv_cache_dtype(True, "fp8_ds_mla",   None) == ("fp8_ds_mla",   torch.uint8)  # no regression
assert get_kv_quant_mode("nvfp4_ds_mla") == KVQuantMode.NVFP4
assert DeepseekSparseSWABackend.get_kv_cache_shape(1, 64, 1, 512, cache_dtype_str="nvfp4_ds_mla") == (1, 64, 584)
assert _swa_page("nvfp4_ds_mla") == _swa_page("fp8_ds_mla")   # the invariant
```

The last one caught the page-size divergence at build time, printing
`32832 vs 37440` instead of costing another boot.

## 4. Why it is not a better production config

Measured against the shipped `fp8_ds_mla` recipe on the same image, model and
hardware:

| | `fp8_ds_mla` | NVFP4 patch |
|---|---:|---:|
| bytes/token | 11,315 | **8,864 (-22%)** |
| best KV pool | **1,674,044** | 1,507,777 |
| serves | **yes** | no |
| c5 tok/s | **140.2** | never reached |

### The 22% saving is not real. It is under-allocation.

An earlier version of this document called the saving "confirmed three
independent ways". That was wrong: the measured pool, the engine's 8.66 GiB
floor and the allocator arithmetic are not independent, they are all readings of
the same page-size computation. If that computation is wrong, all three agree
and all three are wrong.

Two pieces of evidence that `nvfp4_ds_mla` costs the **same** bytes per token as
`fp8_ds_mla`:

- Direct measurement in `TUNING.md`: fp8 **9,094** B/token, nvfp4_ds_mla
  **9,083** B/token. Identical.
- Both independent implementations of the format (ours and anemll's) route it
  through the same **584-byte** envelope. The encoding differs, the slot size
  does not. RoPE stays bf16 either way, so a 2x saving was never on the table.

Reconstructing the 8,864 figure: with `A` the bytes/token from the main
compressed-MLA group and `B` everything else,

```
 A + B = 11,315   (fp8)
xA + B =  8,864   (nvfp4)
```

`x = 432/584 = 0.7397` gives `A = 9,416`, and at `compress_ratio=4` a 584-byte
slot costs `584/4 = 146` B/token/layer, so `A/146 = 64.5` layers. That is a
clean fit for a ~64-layer model. **At least one KV group is sized with a
432-byte generic-NVFP4 page while the writer still emits 584 bytes**, a 26%
under-allocation.

That explains the failure signature exactly: it allocates cleanly, survives a
trivial forward pass, then dies once capture or warmup touches enough pages.

**Why the build assertion missed it.** `_swa_page("nvfp4_ds_mla") ==
_swa_page("fp8_ds_mla")` covers one spec. There are **29** exact-string
`== "fp8_ds_mla"` comparisons across the DSv4/MLA path in v0.27.1, and this
patch touches 5. The correct assertion iterates every KV cache group:

```python
for spec_fp8, spec_nv in zip(specs("fp8_ds_mla"), specs("nvfp4_ds_mla")):
    assert spec_fp8.page_size_bytes == spec_nv.page_size_bytes
```

If every group matches, NVFP4 offers no per-token saving on this model and this
line of work is finished. If any group differs, that difference is the bug.
Either result is worth more than another utilization sweep.

### Runs, in order

| util | capture | cudagraph | result |
|---|---|---|---|
| 0.85 | 64 | `FULL_AND_PIECEWISE` | KV 1,500,286; OOM in PIECEWISE capture |
| 0.82 | 36 | `FULL_AND_PIECEWISE` | pool below the 8.66 GiB floor; refused to start |
| 0.85 | 36 | `FULL_AND_PIECEWISE` | KV 1,400,955; OOM in PIECEWISE capture |
| 0.85 | 36 | `FULL` | KV 1,507,777; still exhausts memory |

`max_cudagraph_capture_size` only needs to cover `max_num_seqs x (k+1)` = 36;
the inherited 64 captured 28 unreachable sizes. Reducing it increased the KV
pool, which confirms capture memory is reserved before KV is sized.

## 5. Relationship to upstream

eugr PR [#311](https://github.com/eugr/spark-vllm-docker/pull/311)
("DeepSeek V4 Flash DSpark C12 NVFP4 recipe + mod", open since 2026-07-07) also
targets NVFP4 for this model, but differently: **+25,079 lines across 23 files**,
wrapping the third-party `vllm-dspark-runtime:dspark-nvfp4-stage-c` container as
a mod, capped at 350K context. It does not make eugr's own image NVFP4-capable.

This patch is complementary and fits eugr's existing mod convention
(`mods/<name>/{run.sh,*.patch,README.md}`, applied with
`patch -p1 -d /usr/local/lib/python3.12/dist-packages`). It is not proposed as a
production recipe until §4 is resolved.

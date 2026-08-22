# Why KV offload to disk fails for DeepSeek-V4 (MLA)

Enabling `OffloadingConnector` with an `fs_python` disk tier on DeepSeek-V4 kills
the worker during startup:

```
RuntimeError: Worker failed with error 'CUDA error: an illegal memory access
was encountered'
```

Reproduced on vLLM 0.27.1 **with and without** the b12x attention backend, so it
is not a b12x interaction. On an older stack (0.21.1rc1) the same configuration
deadlocked in CUDA-graph capture instead of faulting.

This is a **layout mismatch in the offload worker**, not a configuration error.
No flag fixes it.

---

## The offload path assumes a flat, unpadded 2-D int8 cache

`vllm/v1/kv_offload/cpu/gpu_worker.py` asserts exactly that:

```python
for gpu_tensor, cpu_tensor in zip(gpu_tensors, cpu_tensors):
    assert gpu_tensor.dtype == torch.int8
    assert gpu_tensor.ndim == 2
    ...
    _, gpu_page_size = gpu_tensor.shape
    _, cpu_page_size = cpu_tensor.shape
    assert cpu_page_size == gpu_page_size * blocks_per_chunk
```

and then computes transfer destinations with **raw pointer arithmetic**, bypassing
the tensor API:

```python
# compute_sub_block_ptrs()
base_ptr = tensor.data_ptr()
row_stride = tensor.stride(0)
...
block_page_size = tensor.shape[1] // blocks_per_chunk
all_ptrs = (base_ptr + block_ids * row_stride) + sub_offsets
```

Those addresses go to a Triton kernel that reinterprets them as `int64*`:

```python
# swap_blocks_triton.py
src = tl.load(src_addrs + job).to(tl.pointer_type(tl.int64))
words = tl.load(sizes + job) // 8
data = tl.load(src + idx, mask=idx < words, other=0)
```

**Nothing validates that the addresses lie inside the buffer.** The `mask` bounds
the copy against the *declared* size, not against the allocation. A wrong
`row_stride` or `block_page_size` reads or writes out of bounds with no check,
which surfaces as an illegal memory access, or corrupts memory silently.

## Why DeepSeek-V4 violates the assumption

Its MLA cache is a **padded uint8 page layout**, not a flat int8 matrix. From
`models/deepseek_v4/attention.py`:

```python
alignment = 584 if nvfp4_ds_mla else 576 if fp8_ds_mla else 512
```

So:

| Assumption in offload worker | DeepSeek-V4 MLA reality |
|---|---|
| `dtype == torch.int8` | `uint8` packed pages |
| page bytes = `shape[1] // blocks_per_chunk` | `MLAAttentionSpec.page_size_bytes`, with 576/584-byte alignment padding |
| one uniform block geometry | sparse MLA adds a second group (lightning indexer) with its own geometry |
| flat 2-D rows | paged layout whose stride includes alignment padding |

Once `block_page_size` diverges from the true padded page size, every computed
pointer drifts by an accumulating offset.

## The only MLA-awareness that exists is host-side

`vllm/v1/kv_offload/config.py` has a `replicated_layout` flag, described as:

> True when the offloaded bytes of every worker are expected to be byte-identical
> per block (pure-MLA model, single-node TP-only parallelism), enabling a
> single-copy host layout in backends that support it.

It is guarded carefully:

```python
replicated_layout = (
    vllm_config.model_config.use_mla
    and type(single_group_spec) is MLAAttentionSpec   # exact type, fail closed
    and worker_kv_bytes_per_block
        == single_group_spec.page_size_bytes * len(layer_names)
    and parallel_config.tensor_parallel_size > 1
    and parallel_config.pipeline_parallel_size == 1
    ...
)
```

That is purely a **host memory layout** optimization (one copy instead of N).
It says nothing about GPU-side page geometry, and `kv_offload/cpu/gpu_worker.py`
never consults `MLAAttentionSpec`.

Note `worker_kv_bytes_per_block` itself is derived correctly, from the real
allocation:

```python
worker_kv_bytes_per_block = total_gpu_kv_bytes // kv_cache_config.num_blocks
```

so the *aggregate* sizing is right. The failure is in per-block **addressing**.

## What is lacking

1. **MLA page geometry in the offload worker.** Per-block offsets must come from
   `MLAAttentionSpec.page_size_bytes` / `alignment`, not `tensor.shape[1]`.
2. **Bounds validation before the copy.** Computed pointers are never checked
   against `data_ptr() + numel() * element_size()`. This is the difference
   between a clear error and silent corruption, and it is cheap: a
   host-side assert over the pointer array, once per batch.
3. **Multi-group support.** Sparse MLA has more than one KV cache group with
   differing geometry; the transfer path assumes uniformity.
4. **dtype breadth.** The `int8` assertion excludes the `uint8` packed layouts
   that MLA models actually use.

Item 2 is worth upstreaming on its own, independently of MLA: any layout
mismatch in this path currently produces an illegal memory access rather than a
diagnosable failure, and an IMA is the *lucky* outcome. The unlucky one is a
successful copy of the wrong bytes.

**Submitted upstream as
[vllm-project/vllm#53271](https://github.com/vllm-project/vllm/pull/53271)**
("[Bugfix][KV offload] Validate computed device pointers before copy"), with the
patch kept here as
[`patches/kv-offload-bounds-check.patch`](patches/kv-offload-bounds-check.patch).
It adds a host-side bounds check on both exit paths of
`compute_sub_block_ptrs()`: one min/max over the pointer array per batch, raising
a `ValueError` that names the offending range, tensor shape, dtype and stride.
It deliberately does not attempt items 1, 3 and 4, which are a feature rather
than a bug fix.

## Tested end to end, 2026-08-22

The analysis above was static. It has since been built and run on 2x DGX Spark
(GB10, TP=2) against the stock eugr b12x image, using the `fs` (filesystem)
secondary tier on NVMe. Both KV layouts fail identically.

Config, with the schema taken from `TieringOffloadingSpec`'s own docstring
rather than guessed:

```
--kv-transfer-config '{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "TieringOffloadingSpec",
    "cpu_bytes_to_use": 8589934592,
    "secondary_tiers": [{
      "type": "fs",
      "root_dir": "/cache/kvspill",
      "n_read_threads": 8, "n_write_threads": 8,
      "gc_max_size_gb": 512, "gc_low_watermark": 0.85
    }]
  }
}'
```

| KV dtype | page layout | result |
|---|---|---|
| `fp8_ds_mla` | 584B padded | `CUDA error: an illegal memory access was encountered` |
| `fp8` | flat rows | **identical fault** |

Two findings that correct assumptions in the sections above:

**The offload machinery works; only the transfer path fails.** The connector is
constructed and validated (it rejects `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`
by name, because offload registers fixed DMA buffers that expandable segments
cannot provide), and **~1 MB of KV reached the NVMe before the fault**. So this
is not a configuration or capability gap, it is `compute_sub_block_ptrs()`
walking off the allocation exactly as predicted.

**Switching to a flat-row dtype does not help**, which rules out the obvious
workaround. `fp8` satisfies the flat-layout assumption for the *main* MLA group,
but DeepSeek-V4 builds a hybrid cache with three groups (main MLA at
`compress_ratio` 4/128, the sparse-indexer cache, and a sliding-window group).
The indexer and SWA groups take `_opaque_fallback_mapping` regardless of the
main dtype, so item 3 above (multi-group support) is the binding blocker, not
item 1 (page geometry) as originally supposed.

Practical consequence: **there is no KV-dtype or config combination that enables
disk offload for DeepSeek-V4 on vLLM 0.27.1.** Anyone attempting this should
start from item 3.

## Why this repo does not work around it

Forcing the transfer would mean writing KV pages the kernels then read back with
different geometry. That class of bug can produce wrong tokens instead of a
crash, which is worse than having no offload. The capability is left disabled and
documented rather than patched blind.

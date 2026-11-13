# Phase 0: Hardware Topology & TorchTPU Ecosystem API Discovery

- **Generated Timestamp (UTC)**: `2026-11-10T05:06:42.005854+00:00`
- **Host**: `rkaranjai-nanodit-dev` (Python `3.12.15`)
- **TPU Device Kind**: `TPU v7 (tpu7x-16)` (`chip_count=4`, `HBM3E/chip=192.0 GiB`, `total_HBM3E=768.0 GiB`)
- **VFIO Nodes**: `/dev/vfio/0, /dev/vfio/1, /dev/vfio/2, /dev/vfio/3, /dev/vfio/4, /dev/vfio/5, /dev/vfio/6, /dev/vfio/7`

## 1. Installed Package Versions

| Component | Installed Version |
| :--- | :--- |
| `torch` | `2.13.0+cpu` |
| `torch_tpu` | `0.2.0` |
| `vllm` | `0.31.0+empty` |
| `vllm_torchtpu` | `0.0.0` |
| `jax` | `0.11.2` |
| `pallas` | `0.11.2` |
| `tokamax` | `0.0.17` |
| `tpu_info` | `0.7.1` |
| `xprof` | `2.24.2` |
| `datasets` | `5.2.0` |
| `seaborn` | `0.13.2` |

## 2. TPU v7 Device Matmul Verification (`BF16` & `FP8`)

- **Shape `(M, K, N)`**: `(1024, 2048, 2048)`
- **BF16 `torch.compile(backend='tpu')`**: `91.56 TFLOPS` (`0.0938 ms`, cosine sim vs FP32 = `0.999977`)
- **FP8 Dispatch Op**: `torch._scaled_mm`
- **FP8 `torch.compile(backend='tpu')`**: `94.53 TFLOPS` (`0.0909 ms`, cosine sim vs FP32 = `0.999272`)

## 3. `torch.tpu.pallas.jax_op` & Custom Kernel Entry Points

- **Module Path**: `<builtin>`
- **Discovered Callable Signatures**:
```python
torch.tpu.pallas.get_pallas_mesh(axis_names: collections.abc.Sequence[str] | str, mesh_shape: collections.abc.Sequence[int] | None = None, process_group: torch.distributed.distributed_c10d.ProcessGroup | None = None) -> jax._src.mesh.Mesh
torch.tpu.pallas.jax_op(name, fn=None, /, *, donate_argnums=None, mesh: jax._src.mesh.Mesh | None = None, input_partition_specs=None, output_partition_specs=None)
```

## 4. `torch.tpu.profiler` Entry Points

- **Module Path**: `<builtin>`
```python
torch.tpu.profiler.TpuProfilerConfig(*, host_tracer_level: 'int' = 2, device_tracer_level: 'int' = 1, python_tracer_level: 'int' = 0, run_dir: 'os.PathLike[str] | str | None' = None, worker_rank: 'str | None' = None, experimental_options: 'Mapping[str, _ExperimentalValue] | None' = None, check_experimental_options: 'bool' = True)
```

## 5. `openxla/tokamax` Discovery

- **Module Root**: `/usr/local/lib/python3.12/site-packages/tokamax`
- **Top-Level Exports**: `AutotuningResult, BatchedShapeDtype, BenchmarkData, BoundArguments, DISABLE_JAX_EXPORT_CHECKS, DotProductAttentionImplementation, Op, RaggedDotGroupSizes, autotune, autotuning, benchmark, benchmarking, config, dot_product_attention, gated_linear_unit, generate_ragged_dot_group_sizes, layer_norm, linear_softmax_cross_entropy_loss, ops, ragged_dot, ragged_dot_general, snake_case_name, standardize_function, triangle_multiplication`

## 6. `vllm` / `vllm-torchtpu` Speculative Decoding Entry Points

- **`SpeculativeConfig` Fields**: `enforce_eager, num_speculative_tokens, model, method, draft_tensor_parallel_size, tensor_parallel_size, quantization, moe_backend, attention_backend, kv_cache_dtype, max_model_len, revision, code_revision, index_share_for_mtp_iteration, disable_padded_drafter_batch, disable_eagle_block_drop, use_local_argmax_reduction, use_heterogeneous_vocab, prompt_lookup_max, prompt_lookup_min, parallel_drafting, target_model_config, target_parallel_config, num_speculative_tokens_per_batch_size, draft_model_config, draft_parallel_config, suffix_decoding_max_tree_depth, suffix_decoding_max_cached_requests, suffix_decoding_max_spec_factor, suffix_decoding_min_token_prob, draft_load_config, rejection_sample_method, synthetic_acceptance_rates, synthetic_acceptance_length, enable_adaptive_verification, draft_sample_method, dspark_draft_topk`

### Discovered Source Files Matching `DFlash` / `DSpark` / `EAGLE-3` / `ngram` / `jax_op`

| Package | Relative Path | Matched Capabilities |
| :--- | :--- | :--- |
| `vllm_torchtpu` | `vllm_torchtpu/__init__.py` | `dflash, dspark, eagle` |
| `vllm_torchtpu` | `vllm_torchtpu/compilation/tpu_compiler.py` | `eagle, jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/core/eagle_block_drop.py` | `dspark, eagle` |
| `vllm_torchtpu` | `vllm_torchtpu/distributed/pp_shift.py` | `jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/distributed/sharded_jax_op.py` | `jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/kernels/deepseek_v4/core_attention/mla_swa.py` | `dflash, dspark` |
| `vllm_torchtpu` | `vllm_torchtpu/kernels/experimental/adaptive_fused_moe/vllm_adapter.py` | `jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/kernels/experimental/pcp_streaming_rpa/vllm_adapter.py` | `jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/kernels/partial_kv_update.py` | `jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/kernels/router_topk.py` | `jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/layers/adapter/attention.py` | `dflash, eagle3, eagle, jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/layers/adapter/cp_attention.py` | `jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/layers/adapter/custom_ops/deepseek_v4/deepseek_v4_attention_op.py` | `jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/layers/adapter/custom_ops/deepseek_v4/deepseek_v4_compressor.py` | `jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/layers/adapter/custom_ops/deepseek_v4/deepseek_v4_indexer.py` | `jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/layers/adapter/custom_ops/deepseek_v4/deepseek_v4_mhc_op.py` | `jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/layers/adapter/custom_ops/gdn_attention_op.py` | `jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/layers/adapter/custom_ops/kda_attention_op.py` | `jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/layers/adapter/custom_ops/mamba_state_copy_op.py` | `jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/layers/adapter/custom_ops/mla_attention_op.py` | `jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/layers/adapter/custom_ops/optimization_barrier_op.py` | `jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/layers/adapter/fused_moe.py` | `jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/layers/adapter/fused_moe_decode.py` | `jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/layers/adapter/fused_moe_ep.py` | `jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/layers/adapter/linear_common.py` | `jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/layers/adapter/moe_tp_decode.py` | `jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/layers/adapter/quantization/fp8.py` | `jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/layers/adapter/router_topk.py` | `jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/layers/adapter/vision_attention.py` | `jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/models/vllm/__init__.py` | `dspark` |
| `vllm_torchtpu` | `vllm_torchtpu/models/vllm/deepseek_v4/attention.py` | `dflash, dspark, jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/models/vllm/deepseek_v4/dspark.py` | `dflash, dspark` |
| `vllm_torchtpu` | `vllm_torchtpu/models/vllm/kimi_k3/__init__.py` | `dspark` |
| `vllm_torchtpu` | `vllm_torchtpu/models/vllm/kimi_k3/collective_ops.py` | `jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/models/vllm/kimi_k3/dspark.py` | `dspark` |
| `vllm_torchtpu` | `vllm_torchtpu/models/vllm/kimi_k3/layers.py` | `jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/models/vllm/vllm_model_wrapper_context.py` | `jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/offload/block_major_layout.py` | `jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/omni/attention.py` | `jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/omni/rope.py` | `jax_op` |
| `vllm_torchtpu` | `vllm_torchtpu/patch_registry.py` | `dflash, eagle` |
| `vllm_torchtpu` | `vllm_torchtpu/runner/speculative_decoding_manager.py` | `dflash, dspark, eagle3, eagle, ngram` |
| `vllm_torchtpu` | `vllm_torchtpu/runner/tpu_runner.py` | `dflash, dspark, eagle3, eagle, ngram` |
| `vllm_torchtpu` | `vllm_torchtpu/spec_decode/dflash.py` | `dflash, dspark, eagle3, eagle` |
| `vllm_torchtpu` | `vllm_torchtpu/spec_decode/dspark.py` | `dflash, dspark, eagle` |
| `vllm_torchtpu` | `vllm_torchtpu/spec_decode/eagle3.py` | `dflash, dspark, eagle3, eagle` |
| `vllm_torchtpu` | `vllm_torchtpu/spec_decode/mtp_fused_loop.py` | `eagle3, eagle` |
| `vllm_torchtpu` | `vllm_torchtpu/spec_decode/mtp_loop_model.py` | `eagle3, eagle` |
| `vllm_torchtpu` | `vllm_torchtpu/spec_decode/utils.py` | `dflash, dspark, eagle3, eagle` |
| `vllm` | `vllm/config/engram.py` | `ngram` |
| `vllm` | `vllm/config/kernel.py` | `dspark` |
| `vllm` | `vllm/config/speculative.py` | `dflash, dspark, eagle3, eagle, ngram` |
| `vllm` | `vllm/config/vllm.py` | `dflash, dspark, eagle3, eagle, ngram` |
| `vllm` | `vllm/envs.py` | `dflash` |
| `vllm` | `vllm/model_executor/layers/attention/mla_attention.py` | `dspark` |
| `vllm` | `vllm/model_executor/models/cohere_eagle.py` | `eagle` |
| `vllm` | `vllm/model_executor/models/deepseek_eagle.py` | `eagle` |
| `vllm` | `vllm/model_executor/models/deepseek_eagle3.py` | `eagle3, eagle` |
| `vllm` | `vllm/model_executor/models/eagle2_5_vl.py` | `eagle` |
| `vllm` | `vllm/model_executor/models/gemma4_dspark.py` | `dflash, dspark` |

### Relevant `vllm` CLI Flags (`--help` Excerpt)

```text
                     [--quantization QUANTIZATION]
                     [--quantization-config QUANTIZATION_CONFIG]
                     [--allow-deprecated-quantization | --no-allow-deprecated-quantization]
                     [--kv-cache-dtype {auto,bfloat16,float16,fp8,fp8_ds_mla,fp8_e4m3,fp8_e5m2,fp8_inc,fp8_per_token_head,int4_per_token_head,int8_per_token_head,nvfp4,nvfp4_4over6,nvfp4_ds_mla,turboquant_3bit_nc,turboquant_4bit_nc,turboquant_k3v4_nc,turboquant_k8v4}]
                     [--kv-cache-dtype-skip-layers KV_CACHE_DTYPE_SKIP_LAYERS [KV_CACHE_DTYPE_SKIP_LAYERS ...]]
                     [--async-scheduling | --no-async-scheduling]
                     [--speculative-config SPECULATIVE_CONFIG]
```

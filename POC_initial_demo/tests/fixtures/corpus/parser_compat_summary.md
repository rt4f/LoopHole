# Parser Corpus Compatibility Summary

- Corpus root: `C:\Users\ADMIN\Desktop\LoopHole\POC_initial_demo\tests\fixtures\corpus`
- Total files: 6
- Compatible files: 2
- Incompatible files: 4

## Incompatibility Class Frequency

| Class | Frequency |
|---|---:|
| no_loop_construct_detected | 2 |
| no_store_detected | 2 |
| affine_apply_index_materialization | 1 |
| conditional_region_not_modeled | 1 |
| scf_iter_args_not_modeled | 1 |

## File-Level Results

| File | Compatible | Classes | Loops | Reads | Writes | Diagnostics |
|---|---|---|---:|---:|---:|---:|
| frontend_no_loop_vectorized.mlir | no | no_loop_construct_detected, no_store_detected | 0 | 0 | 0 | 3 |
| polygeist_affine_apply_indices.mlir | no | affine_apply_index_materialization | 1 | 1 | 1 | 1 |
| polygeist_conv1d_scf.mlir | yes | - | 2 | 3 | 1 | 0 |
| polygeist_dot_iter_args.mlir | no | no_loop_construct_detected, no_store_detected, scf_iter_args_not_modeled | 0 | 2 | 0 | 4 |
| polygeist_if_guarded_store.mlir | no | conditional_region_not_modeled | 1 | 1 | 1 | 1 |
| polygeist_matmul_affine.mlir | yes | - | 3 | 3 | 1 | 0 |

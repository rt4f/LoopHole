# YYYY-MM-DD — Experiment title

- **Owner:**
- **Status:** planned | running | done | abandoned
- **Related:** ADR / plan / issue links

## Question and hypothesis

What do we want to learn, and what do we expect to see (written before running)?

## Setup

| Item | Value |
|---|---|
| Commit | `git rev-parse HEAD` |
| Docker image | `loophole-polygeist:llvm17` (image ID: `docker images -q loophole-polygeist:llvm17`) |
| Toolchain refs | from `docker/polygeist/toolchain.lock` |
| Machine | CPU / RAM / GPU / OS |
| Inputs | fixture set, kernels, parameters |

## Procedure

```bash
# exact commands, copy-pasteable
```

## Results

Summary table / plot (files in `results/`). Raw outputs are in `outputs/` (not committed).

## Conclusion

Was the hypothesis supported? What changes (code, ADR, baseline, paper)?

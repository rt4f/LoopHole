# tools/legacy/ (unsupported)

Superseded tooling kept for reference. **The supported path is the Docker image;
see [SETUP.md](../../SETUP.md).** Nothing here is tested or maintained, and the
scripts still reference pre-restructure paths (`POC_initial_demo/`, `scripts/`),
so they will not run as-is.

| Folder | What it was | Superseded by |
|---|---|---|
| [docker-source-build/](docker-source-build/) | Dockerfile that builds all of LLVM/MLIR/Clang + Polygeist from source (multi-hour). Was `docker/` with build context at the repo root; its `.dockerignore` is kept alongside. | `docker/polygeist/` (apt LLVM 17 + `cgeist` build, ~5 min) |
| [wsl-native-build/](wsl-native-build/) | WSL2 scripts to build LLVM + Polygeist natively, plus `INSTALL.txt` (was `llvm_install_instruction.txt`). | Docker image |

To revive one, move it out of `legacy/`, fix its paths, add it to CI, and write an ADR.

# C-01 and C-02 Implementation Details

Date: 2026-04-03
Scope: Demo reliability and CI smoke coverage

## C-01: Standalone demo compatibility

Primary file:
1. `POC_initial_demo/examples/run_demo.py`

Key updates:
1. Added missing imports required by runtime paths.
2. Replaced stale LiftResult field usage with current `emitted_mlir`-based flow.
3. Added helper logic for emitted op extraction across linalg and stablehlo text.
4. Added environment-driven controls:
   - `LOOPHOLE_DEMO_FIXTURE_LIMIT`
   - `LOOPHOLE_DEMO_Z3_TIMEOUT_MS`
5. Added terminal-safe output rendering to avoid Windows cp1252 UnicodeEncodeError during demo output.

Windows-specific reliability fix:
1. Prior behavior could crash on cp1252 terminals when Unicode characters appeared in summary or emitted MLIR output.
2. Runtime output is now encoded safely for the active terminal encoding before printing.
3. Status arrows in user-visible output were normalized to ASCII for consistency.

## C-02: Demo smoke tests

Primary file:
1. `POC_initial_demo/tests/integration/test_demo_script_smoke.py`

Test design:
1. Run script in plain mode and assert process return code is zero.
2. Run script in demo mode and assert process return code is zero.
3. Fail test if traceback text appears in combined stdout/stderr.
4. Keep execution budget small with fixture-limit and timeout env vars to make CI-friendly smoke tests.

Why integration smoke was necessary:
1. Unit tests do not validate entrypoint behavior, terminal rendering, or command-line path wiring.
2. Demo regressions are high-visibility failures for contributors and evaluators.
3. This provides fast guardrails against stale field refactors and runtime output encoding regressions.

## Acceptance criteria mapping

C-01 done when:
1. Standalone demo runs in rich path and plain path without runtime attribute errors.
2. Demo can render emitted output using current LiftResult model.

C-02 done when:
1. CI-executable smoke tests cover both modes.
2. Smoke tests consistently pass in the project venv.

## Verification commands

1. `c:/Users/Yasho/LoopHole/.venv/Scripts/python.exe -m pytest tests/integration/test_demo_script_smoke.py -q`
2. Optional local run: `c:/Users/Yasho/LoopHole/.venv/Scripts/python.exe examples/run_demo.py --mode plain`
3. Optional local run: `c:/Users/Yasho/LoopHole/.venv/Scripts/python.exe examples/run_demo.py --mode demo`

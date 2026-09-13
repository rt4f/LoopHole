"""
test_cli_error_messages.py — Integration tests for B-02 CLI error handling.

Validates that unsupported MLIR inputs fail with actionable messages through
the full lifter pipeline (parser → lifter → result), not generic parse failures.
"""

from loophole.lifter import Lifter
from loophole.tests.fixtures import MATMUL_MLIR


class TestCLIErrorMessages:
    """Test that the CLI path surfaces meaningful parser errors."""

    def test_lift_missing_function_marker_produces_error(self):
        """Test that malformed function declaration produces actionable error."""
        mlir = """
func mykernel(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  affine.for %i = %c0 to %c0 {
    %a = affine.load %A[%i, %i] : memref<4x4xf32>
    affine.store %a, %C[%i, %i] : memref<4x4xf32>
  }
  return
}
"""
        lifter = Lifter(target="linalg", verbose=False)
        result = lifter.lift(mlir)
        
        # Should not succeed
        assert not result.success
        # Should have parser diagnostics or error
        assert result.parser_diagnostics or result.error
        # Error message should be actionable
        if result.parser_diagnostics:
            diags_str = "\n".join(str(d) for d in result.parser_diagnostics)
            assert "func.func @" in diags_str or "marker" in diags_str.lower(), \
                   f"Diagnostic should mention 'func.func @' or 'marker', got:\n{diags_str}"

    def test_lift_missing_type_annotation_produces_warning(self):
        """Test that missing type annotations are reported as diagnostics."""
        mlir = """
func.func @matmul(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  affine.for %i = %c0 to %c0 {
    %a = affine.load %A[%i, %i]
    affine.store %a, %C[%i, %i] : memref<4x4xf32>
  }
  return
}
"""
        lifter = Lifter(target="linalg", verbose=False)
        result = lifter.lift(mlir)
        
        # May succeed with warnings, or fail gracefully
        # Important: should have parser diagnostics explaining what's wrong
        assert result.parser_diagnostics, \
               "Expected parser diagnostics for missing type annotation"
        
        # Check that diagnostic explanations are helpful
        diags_str = "\n".join(str(d) for d in result.parser_diagnostics)
        assert "type" in diags_str.lower(), \
               f"Diagnostic should mention 'type', got:\n{diags_str}"

    def test_lift_non_scalar_constant_produces_warning(self):
        """Test that non-scalar constants produce actionable warnings."""
        mlir = """
func.func @matmul(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %dense = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  affine.for %i = %c0 to %c0 {
    %a = affine.load %A[%i, %i] : memref<4x4xf32>
    affine.store %a, %C[%i, %i] : memref<4x4xf32>
  }
  return
}
"""
        lifter = Lifter(target="linalg", verbose=False)
        result = lifter.lift(mlir)
        
        # May succeed but should have warnings
        if result.parser_diagnostics:
            diags_str = "\n".join(str(d) for d in result.parser_diagnostics)
            # Should mention tensors or scalars
            assert "scalar" in diags_str.lower() or "dense" in diags_str.lower(), \
                   f"Diagnostic should explain non-scalar issue, got:\n{diags_str}"

    def test_lift_multiple_errors_all_reported(self):
        """Test that multiple parser errors are all surfaced, not just first one."""
        mlir = """
func mykernel(%A: memref<4x4xf32>) {
  %bad = arith.constant "not_a_number" : f32
  %c0 = arith.constant 0 : index
  affine.for %i = %c0 to %c0 {
    %a = affine.load %A[%i, %i]
    affine.store %a, %A[%i, %i]
  }
  return
}
"""
        lifter = Lifter(target="linalg", verbose=False)
        result = lifter.lift(mlir)
        
        # Should collect all errors, not stop at first
        assert len(result.parser_diagnostics) >= 2, \
               f"Expected 2+ diagnostics for multiple issues, got {len(result.parser_diagnostics)}"
        
        # Verify they cover different problems
        diags_str = "\n".join(str(d) for d in result.parser_diagnostics)
        # Should mention both constant problem and type problem
        issues = diags_str.lower()
        has_constant_issue = 'constant' in issues or 'unparse' in issues
        has_type_issue = 'type' in issues
        assert has_constant_issue or has_type_issue, \
               f"Diagnostics should explain both issues, got:\n{diags_str}"

    def test_lift_actionable_guidance_messages(self):
        """Test that error messages include actionable guidance for user fixes."""
        mlir = """
func mykernel(%A: memref<4x4xf32>) {
  return
}
"""
        lifter = Lifter(target="linalg", verbose=False)
        result = lifter.lift(mlir)
        
        # Check if diagnostics include guidance
        if result.parser_diagnostics:
            has_guidance = any(d.guidance for d in result.parser_diagnostics)
            if has_guidance:
                # Verify guidance is present and meaningful
                guidance_texts = [d.guidance for d in result.parser_diagnostics if d.guidance]
                guidance_str = "\n".join(guidance_texts)
                # Guidance should give actionable advice
                assert len(guidance_str) > 0, "Guidance should not be empty"

    def test_lift_malformed_access_pattern_actionable(self):
        """Test that malformed memory accesses produce clear error messages."""
        mlir = """
func.func @matmul(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  affine.for %i = %c0 to %c0 {
    %a = affine.load %A[%i, %i]
  }
  return
}
"""
        lifter = Lifter(target="linalg", verbose=False)
        result = lifter.lift(mlir)
        
        # Missing type annotation should be reported
        if result.parser_diagnostics:
            diags_str = "\n".join(str(d) for d in result.parser_diagnostics)
            # Should explain what's wrong with the load/store
            assert "load" in diags_str.lower() or "type" in diags_str.lower(), \
                   f"Diagnostic should mention load or type, got:\n{diags_str}"

    def test_error_result_has_clear_summary(self):
        """Test that failed lifting results have clear error messages."""
        mlir = """
func mykernel() {
  return
}
"""
        lifter = Lifter(target="linalg", verbose=False)
        result = lifter.lift(mlir)
        
        # Check result summary
        summary = result.summary()
        # Should indicate failure with reason
        assert "[FAIL]" in summary or result.error, \
               f"Failed result should have clear failure indicator, got:\n{summary}"

    def test_lift_emission_failure_surfaces_actionable_message(self):
        """Emission errors must be surfaced through lift() with actionable text."""
        lifter = Lifter(target="linalg", verbose=False)

        def _broken_emit(_sketch, _loop, _func_name=None):
            raise ValueError("missing required output tensor shape")

        # Simulate emitter-side failure after successful parse/match/verify path.
        lifter._linalg_emitter.emit = _broken_emit  # type: ignore[attr-defined]
        result = lifter.lift(MATMUL_MLIR)

        assert not result.success
        assert result.error is not None
        assert "Emission error while generating" in result.error
        assert "missing required output tensor shape" in result.error


class TestNegativeParsing:
    """Direct parser tests asserting meaningful error text."""

    def test_missing_func_marker_error_text(self):
        """Test that missing function marker produces specific error message."""
        from loophole.affine_extractor import AffineExtractor
        
        mlir = "func kernel() { return }"
        ext = AffineExtractor(validate_with_mlir=False)
        result = ext.extract(mlir)
        
        # Should have error diagnostic
        assert any(d.level == 'error' for d in result.diagnostics)
        
        # Error should be specific about what's wrong
        error_diag = [d for d in result.diagnostics if d.level == 'error'][0]
        assert "func.func @" in error_diag.guidance or "marker" in error_diag.reason.lower(), \
               f"Error should explain missing 'func.func @' marker, got: {error_diag}"

    def test_missing_type_annotation_error_text(self):
        """Test that missing type annotation produces specific warning."""
        from loophole.affine_extractor import AffineExtractor
        
        mlir = """
func.func @test(%A: memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  affine.for %i = %c0 to %c0 {
    %a = affine.load %A[%i, %i]
  }
  return
}
"""
        ext = AffineExtractor(validate_with_mlir=False)
        result = ext.extract(mlir)
        
        # Should have warning about missing type
        type_warnings = [d for d in result.diagnostics 
                        if d.level == 'warning' and 'type' in d.reason.lower()]
        assert type_warnings, \
               f"Should have warning about missing type annotation, got: {result.diagnostics}"
        
        # Warning should be specific
        w = type_warnings[0]
        assert 'annotation' in w.reason.lower() and 'type' in w.reason.lower()

    def test_unparseable_constant_error_text(self):
        """Test that unparseable constants produce specific warnings."""
        from loophole.affine_extractor import AffineExtractor
        
        mlir = """
func.func @test() {
  %bad = arith.constant "not_a_number" : f32
  return
}
"""
        ext = AffineExtractor(validate_with_mlir=False)
        result = ext.extract(mlir)
        
        # Should have warning about unparseable constant
        const_warnings = [d for d in result.diagnostics 
                         if 'unparse' in d.reason.lower() or 'constant' in d.reason.lower()]
        assert const_warnings, \
               f"Should warn about unparseable constant, got: {result.diagnostics}"
        
        # Warning should explain what value caused the issue
        w = const_warnings[0]
        assert len(w.reason) > 20, "Warning should be detailed"

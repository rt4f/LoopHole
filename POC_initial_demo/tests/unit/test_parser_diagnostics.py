"""
test_parser_diagnostics.py — Test parser diagnostic reporting for B-02.

Validates that all silent parser failures are converted to explicit diagnostics.
"""

from loophole.affine_extractor import AffineExtractor


class TestParserDiagnostics:
    """Test that parser failures generate clear diagnostic messages."""

    def test_parse_func_no_marker(self):
        """Test P-03: Missing function marker generates error diagnostic."""
        mlir = """
func mykernel(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  affine.for %i = %c0 to %c1 {
    affine.load %A[%i, %i] : memref<4x4xf32>
  }
  return
}
"""
        ext = AffineExtractor(validate_with_mlir=False)
        result = ext.extract(mlir)
        
        # Expect error diagnostic for missing "func.func @"
        assert any(d.level == 'error' and '_parse_func_declaration' in d.location 
                  for d in result.diagnostics), \
               f"Expected parse error diagnostic, got: {result.diagnostics}"
        assert result.func_name == "unknown"

    def test_parse_func_unclosed_args(self):
        """Test P-03: Unclosed function args generates error diagnostic."""
        mlir = """
func.func @mykernel(%A: memref<4x4xf32>, %B: memref<4x4xf32> {
  %c0 = arith.constant 0 : index
  return
}
"""
        ext = AffineExtractor(validate_with_mlir=False)
        result = ext.extract(mlir)
        
        assert any(d.level == 'error' and '_parse_func_declaration' in d.location 
                  for d in result.diagnostics)

    def test_parse_constant_non_scalar(self):
        """Test P-04: Non-scalar constant generates warning diagnostic."""
        mlir = """
func.func @matmul(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %dense_const = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  affine.for %i = %c0 to %c0 {
    affine.for %j = %c0 to %c0 {
      %a = affine.load %A[%i, %j] : memref<4x4xf32>
      affine.store %a, %C[%i, %j] : memref<4x4xf32>
    }
  }
  return
}
"""
        ext = AffineExtractor(validate_with_mlir=False)
        result = ext.extract(mlir)
        
        # Expect warning for non-scalar constant
        assert any(d.level == 'warning' and 'dense tensor' in d.reason
                  for d in result.diagnostics), \
               f"Expected non-scalar constant warning, got: {result.diagnostics}"

    def test_parse_constant_unparseable(self):
        """Test P-04: Unparseable constant generates warning."""
        mlir = """
func.func @matmul(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
  %bad = arith.constant "not_a_number" : f32
  affine.for %i = %c0 to %c0 {
  }
  return
}
"""
        ext = AffineExtractor(validate_with_mlir=False)
        result = ext.extract(mlir)
        
        assert any(d.level == 'warning' and 'Unparseable' in d.reason
                  for d in result.diagnostics)

    def test_parse_load_missing_type(self):
        """Test P-08: Missing type annotation on load generates warning."""
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
        ext = AffineExtractor(validate_with_mlir=False)
        result = ext.extract(mlir)
        
        # Expect warning for missing type annotation
        assert any(d.level == 'warning' and 'type annotation' in d.reason.lower()
                  for d in result.diagnostics), \
               f"Expected missing type warning, got: {result.diagnostics}"

    def test_parse_store_missing_type(self):
        """Test P-08: Missing type annotation on store generates warning."""
        mlir = """
func.func @matmul(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  affine.for %i = %c0 to %c0 {
    %a = affine.load %A[%i, %i] : memref<4x4xf32>
    affine.store %a, %C[%i, %i]
  }
  return
}
"""
        ext = AffineExtractor(validate_with_mlir=False)
        result = ext.extract(mlir)
        
        assert any(d.level == 'warning' and 'type annotation' in d.reason.lower()
                  for d in result.diagnostics)

    def test_parse_no_tensor_types(self):
        """Test P-15: Default element type when no types discovered."""
        mlir = """
func.func @matmul() {
  %c0 = arith.constant 0 : index
  affine.for %i = %c0 to %c0 {
    %val = arith.addf %c0, %c0 : f32
  }
  return
}
"""
        ext = AffineExtractor(validate_with_mlir=False)
        result = ext.extract(mlir)
        
        # Expect warning for defaulting element type
        assert any(d.level == 'warning' and 'defaulting' in d.reason.lower()
                  for d in result.diagnostics)
        assert result.element_type == "f32"

    def test_parse_no_writes_classify_parallel(self):
        """Test P-16: All IVs classified parallel when no writes."""
        mlir = """
func.func @test() {
  %c0 = arith.constant 0 : index
  affine.for %i = %c0 to %c0 {
    affine.for %j = %c0 to %c0 {
      %sum = arith.addf %c0, %c0 : f32
    }
  }
  return
}
"""
        ext = AffineExtractor(validate_with_mlir=False)
        result = ext.extract(mlir)
        
        # Expect warning for conservative IV classification
        assert any(d.level == 'warning' and 'parallel' in d.reason.lower()
                  for d in result.diagnostics)

    def test_parse_compute_op_operand_truncation(self):
        """Test P-13: N-ary operation truncated to first 2 operands."""
        mlir = """
func.func @matmul(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  affine.for %i = %c0 to %c0 {
    %a = affine.load %A[%i, %i] : memref<4x4xf32>
    %b = affine.load %B[%i, %i] : memref<4x4xf32>
    %c = affine.load %C[%i, %i] : memref<4x4xf32>
    %result = arith.addf %a, %b, %c : f32
    affine.store %result, %C[%i, %i] : memref<4x4xf32>
  }
  return
}
"""
        ext = AffineExtractor(validate_with_mlir=False)
        result = ext.extract(mlir)
        
        # Expect warning about operand truncation
        assert any(d.level == 'warning' and 'truncat' in d.reason.lower()
                  for d in result.diagnostics), \
               f"Expected operand truncation warning, got: {result.diagnostics}"

    def test_multiple_diagnostics_collected(self):
        """Test that multiple diagnostic issues are all collected."""
        mlir = """
func.func @matmul(%A: memref<4x4xf32>) {
  %bad = arith.constant "not_a_number" : f32
  %c0 = arith.constant 0 : index
  affine.for %i = %c0 to %c0 {
    %a = affine.load %A[%i, %i]
    affine.store %a, %A[%i, %i]
  }
  return
}
"""
        ext = AffineExtractor(validate_with_mlir=False)
        result = ext.extract(mlir)
        
        # Should have multiple diagnostics (unparseable constant + missing type annotation)
        assert len(result.diagnostics) >= 2, \
               f"Expected 2+ diagnostics, got {len(result.diagnostics)}: {result.diagnostics}"
        
        levels = [d.level for d in result.diagnostics]
        assert levels.count('warning') >= 2

    def test_verbose_diagnostics_includes_debug(self):
        """Test that verbose_diagnostics=True includes debug-level messages."""
        mlir = """
func.func @matmul(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  affine.for %i = %c0 to %c0 {
    %a = affine.load %A[%i, %i] : memref<4x4xf32>
    %b = affine.load %B[%i, %i] : memref<4x4xf32>
    %sum = arith.addf %a, %b : f32
    affine.store %sum, %C[%i, %i] : memref<4x4xf32>
  }
  return
}
"""
        ext_verbose = AffineExtractor(validate_with_mlir=False, verbose_diagnostics=True)
        result_verbose = ext_verbose.extract(mlir)
        
        ext_normal = AffineExtractor(validate_with_mlir=False, verbose_diagnostics=False)
        result_normal = ext_normal.extract(mlir)
        
        # Verbose mode may have more diagnostics (debug messages)
        # Normal mode filters out debug-level messages
        assert len(result_verbose.diagnostics) >= len(result_normal.diagnostics)

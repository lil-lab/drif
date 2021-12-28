def assert_shape(tensor, expected_shape, name=""):
    assert list(tensor.shape) == list(expected_shape), (
        f"Tensor {name} expected shape: {expected_shape}. Actual shape: {tensor.shape}")

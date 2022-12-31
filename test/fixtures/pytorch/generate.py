import os
import torch
import numpy as np
from collections import OrderedDict


def save(name, fmt, data):
    use_zip = fmt == "zip"
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, f"{name}.{fmt}.pt")
    torch.save(data, path, _use_new_zipfile_serialization=use_zip)


for fmt in ["zip", "legacy"]:
    save("tensors", fmt, [
        torch.tensor([-1.0, 1.0], dtype=torch.float64),
        torch.tensor([-1.0, 1.0], dtype=torch.float32),
        torch.tensor([-1.0, 1.0], dtype=torch.float16),
        torch.tensor([-1, 1], dtype=torch.int64),
        torch.tensor([-1, 1], dtype=torch.int32),
        torch.tensor([-1, 1], dtype=torch.int16),
        torch.tensor([-1, 1], dtype=torch.int8),
        torch.tensor([0, 1], dtype=torch.uint8),
        torch.tensor([-1.0, 1.0], dtype=torch.bfloat16),
        torch.tensor([1 - 1j, 1 + 1j], dtype=torch.complex128),
        torch.tensor([1 - 1j, 1 + 1j], dtype=torch.complex64)
    ])

    save("numpy_arrays", fmt, [
        np.array([-1.0, 1.0], dtype=np.float64),
        np.array([-1.0, 1.0], dtype=np.float32),
        np.array([-1.0, 1.0], dtype=np.float16),
        np.array([-1, 1], dtype=np.int64),
        np.array([-1, 1], dtype=np.int32),
        np.array([-1, 1], dtype=np.int16),
        np.array([-1, 1], dtype=np.int8),
        np.array([0, 1], dtype=np.uint64),
        np.array([0, 1], dtype=np.uint32),
        np.array([0, 1], dtype=np.uint16),
        np.array([0, 1], dtype=np.uint8),
        np.array([0, 1], dtype=np.bool8),
        np.array([1 - 1j, 1 + 1j], dtype=np.complex128),
        np.array([1 - 1j, 1 + 1j], dtype=np.complex64)
    ])

    save("ordered_dict", fmt, OrderedDict([("x", 1), ("y", 2)]))

    transposed_tensor = torch.tensor(
        [[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]]], dtype=torch.int64).permute(2, 0, 1)
    save("noncontiguous_tensor", fmt, transposed_tensor)

    transposed_array = np.transpose(np.array([[1, 2, 3], [4, 5, 6]]))
    save("noncontiguous_numpy_array", fmt, transposed_array)

# Model parameters

save("state_dict_base", "zip", OrderedDict([
    ("conv.weight", torch.ones(2, 3, 2, 2)),
    ("conv.bias", torch.zeros(2)),
]))

save("state_dict_full", "zip", OrderedDict([
    ("base.conv.weight", torch.ones(2, 3, 2, 2)),
    ("base.conv.bias", torch.zeros(2)),
    # Unexpected shape
    ("classifier.layers.0.weight", torch.ones(1, 1)),
    ("classifier.layers.0.bias", torch.zeros(1)),
    # Missing
    # "classifier.layers.1.weight"
    # "classifier.layers.1.bias"
    # Extra
    ("extra.weight", torch.ones(1))
]))

# Zip64
# Source: https://github.com/pytorch/pytorch/blob/master/test/test_serialization.py#L907
big_model = torch.nn.Conv2d(20000, 3200, kernel_size=3)
save("big_model_zip64", "zip", big_model.state_dict())

import os
import torch
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

    save("ordered_dict", fmt, OrderedDict([("x", 1), ("y", 2)]))

    transposed_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64).t()
    save("noncontiguous_tensor", fmt, transposed_tensor)

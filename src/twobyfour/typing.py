import torch
import torch.nn.functional as F
from . import operators as ops

from typing import Self


class Quaternion(torch.Tensor):

    @staticmethod
    def __new__(cls, data, *args, **kwargs):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, *args, **kwargs)

        if data.ndim == 1:
            data = data.unsqueeze(0)

        if data.shape[-1] != 4:
            if data.shape[-1] == 3:
                data = F.pad(data, (1, 0))
            else:
                raise ValueError(
                    f"Last dimension must be valid Quaternion of size 4, got shape {data.shape}"
                )

        return data.as_subclass(cls)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        ret = super().__torch_function__(func, types, args, kwargs)

        if isinstance(ret, cls) and not (ret.ndim >= 2 and ret.shape[-1] == 4):
            return ret.as_subclass(torch.Tensor)

        return ret

    # Arithmetic
    def __add__(self, other) -> Self: ...
    def __radd__(self, other) -> Self: ...
    def __sub__(self, other) -> Self: ...
    def __rsub__(self, other) -> Self: ...
    def __mul__(self, other) -> Self: ...
    def __rmul__(self, other) -> Self: ...
    def __truediv__(self, other) -> Self: ...
    def __rtruediv__(self, other) -> Self: ...
    def __neg__(self) -> Self: ...
    def __pos__(self) -> Self: ...

    # In-place arithmetic
    def __iadd__(self, other) -> Self: ...
    def __isub__(self, other) -> Self: ...
    def __imul__(self, other) -> Self: ...
    def __itruediv__(self, other) -> Self: ...

    # Memory layout
    def contiguous(self, memory_format=torch.contiguous_format) -> Self: ...
    def clone(self, *, memory_format=None) -> Self: ...

    # Device transfer
    def to(self, *args, **kwargs) -> Self: ...
    def cpu(self, memory_format=None) -> Self: ...
    def cuda(self, device=None, non_blocking=False,
             memory_format=None) -> Self: ...

    # Dtype conversion
    def float(self) -> Self: ...
    def double(self) -> Self: ...
    def half(self) -> Self: ...
    def bfloat16(self) -> Self: ...

    # Autograd
    def detach(self) -> Self: ...
    def requires_grad_(self, mode: bool = True) -> Self: ...

    def squaredsumq(self) -> torch.Tensor:
        return ops.quaternion_squares(self)

    def magq(self) -> torch.Tensor:
        return ops.quaternion_magnitude(self)

    def normq(self) -> "Quaternion":
        return ops.quaternion_normalize(self)

    def conjq(self) -> "Quaternion":
        return ops.quaternion_conjugate(self)

    def invq(self) -> "Quaternion":
        return ops.quaternion_inverse(self)

    def dotq(self, other) -> torch.Tensor:
        return ops.quaternion_dot_product(self, other)

    def mulq(self, other) -> "Quaternion":
        return ops.quaternion_multiply(self, other)

    def rmulq(self, other) -> "Quaternion":
        return ops.quaternion_multiply(other, self)

    def applyq(self, point) -> "Quaternion":
        return ops.quaternion_apply(self, point)

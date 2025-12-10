from typing import TYPE_CHECKING, Self

import torch
import torch.nn.functional as F
from torch import Tensor, contiguous_format

from . import operators as ops


def invalid(data: Tensor) -> bool:
    return not (data.ndim >= 2 and data.shape[-1] == 4)


class Quaternion(torch.Tensor):

    @staticmethod
    def __new__(cls, data, *args, **kwargs):
        if not isinstance(data, Tensor):
            data = torch.tensor(data, *args, **kwargs)

        if not torch.is_floating_point(data):
            data = data.to(torch.get_default_dtype())

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

        if isinstance(ret, list):
            return [
                r.as_subclass(Tensor) if isinstance(
                    r, cls) and invalid(r) else r
                for r in ret
            ]

        if isinstance(ret, tuple):
            return tuple(
                r.as_subclass(Tensor) if isinstance(
                    r, cls) and invalid(r) else r
                for r in ret
            )

        elif isinstance(ret, cls) and invalid(ret):
            ret = ret.as_subclass(Tensor)

        return ret

    if TYPE_CHECKING:
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
        def __iadd__(self, other) -> Self: ...
        def __isub__(self, other) -> Self: ...
        def __imul__(self, other) -> Self: ...
        def __itruediv__(self, other) -> Self: ...
        def float(self) -> Self: ...
        def double(self) -> Self: ...
        def half(self) -> Self: ...
        def bfloat16(self) -> Self: ...
        def clone(self, *, memory_format=None) -> Self: ...
        def requires_grad_(self, mode: bool = True) -> Self: ...
        def detach(self) -> Self: ...
        def contiguous(self, memory_format=contiguous_format) -> Self: ...
        def to(self, *args, **kwargs) -> Self: ...
        def cpu(self, memory_format=None) -> Self: ...
        def cuda(self, device=None, non_blocking=False,
                 memory_format=None) -> Self: ...

    def squaredsumq(self) -> Tensor:
        return ops.quaternion_squares(self)

    def magq(self) -> Tensor:
        return ops.quaternion_magnitude(self)

    def normq(self) -> "Quaternion":
        return ops.quaternion_normalize(self)

    def conjq(self) -> "Quaternion":
        return ops.quaternion_conjugate(self)

    def invq(self) -> "Quaternion":
        return ops.quaternion_inverse(self)

    def dotq(self, other) -> Tensor:
        return ops.quaternion_dot_product(self, other)

    def mulq(self, other) -> "Quaternion":
        return ops.quaternion_multiply(self, other)

    def rmulq(self, other) -> "Quaternion":
        return ops.quaternion_multiply(other, self)

    def applyq(self, point) -> "Quaternion":
        return ops.quaternion_apply(self, point)

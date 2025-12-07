import torch
import torch.nn.functional as F
from . import operators as ops


class Quaternion(torch.Tensor):

    @staticmethod
    def __new__(cls, data, *args, **kwargs):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, *args, **kwargs)

        if data.ndim == 1:
            data = data.unsqueeze(0)

        if data.shape[-1] != 4:
            if data.shape == 3:
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

    def squaredsumq(self) -> torch.Tensor:
        return ops.quaternion_squares(self)

    def magq(self) -> torch.Tensor:
        return ops.quaternion_magnitude(self)

    def normq(self):
        return ops.quaternion_normalize(self)

    def conjq(self):
        return ops.quaternion_conjugate(self)

    def invq(self):
        return ops.quaternion_inverse(self)

    def dotq(self, other):
        return ops.quaternion_dot_product(self, other)

    def mulq(self, other):
        return ops.quaternion_multiply(self, other)

    def rmulq(self, other):
        return ops.quaternion_multiply(other, self)

    def applyq(self, point):
        return ops.quaternion_apply(self, point)

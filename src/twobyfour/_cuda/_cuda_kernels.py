# Copyright (c) 2025 Marisha Norcross
#
# This source code is covered by MIT license
# See LICENSE for the full license text.


from typing import cast, Callable

import torch
from torch import Tensor

from .. import _C  # type: ignore
from ..typing import Quaternion


@torch.library.register_fake("twobyfour::quaternion_multiply")
def _(left: Tensor, right: Tensor) -> Tensor:
    torch._check(left.dtype == torch.float)
    torch._check(left.dtype == torch.float)
    torch._check(left.device == right.device)

    return torch.empty_like(left)


def quat_mul(left: Quaternion, right: Quaternion) -> Quaternion:
    leftbc, rightbc = cast(tuple[Tensor, Tensor],
                           torch.broadcast_tensors(left, right))

    q_m = cast(Callable, torch.ops.twobyfour.quaternion_multiply)
    output: Tensor = q_m(leftbc.contiguous(), rightbc.contiguous())
    return Quaternion(output)

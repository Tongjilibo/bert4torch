''' 量化模块
'''

from torch.nn import Linear, Embedding
from torch.nn.parameter import Parameter
from torch import nn
import torch.nn.functional as F
import bz2
import torch
import base64
import ctypes
from typing import List, Union, Dict
import re
from tqdm import tqdm
from functools import partial
from bert4torch.snippets import is_package_available, has_meta_param
from .base import QuantizerBase

try:
    from cpm_kernels.kernels.base import LazyKernelCModule, KernelFunction, round_up

    class Kernel:
        def __init__(self, code: bytes, function_names: List[str]):
            self.code = code
            self._function_names = function_names
            self._cmodule = LazyKernelCModule(self.code)

            for name in self._function_names:
                setattr(self, name, KernelFunction(self._cmodule, name))

    quantization_code = "$QlpoOTFBWSZTWU9yuJUAQHN//////////f/n/8/n///n//bt4dTidcVx8X3V9FV/92/v4B7/AD5FBQFAAAChSgKpFCFAFVSigUAAAEKhSgUUqgFBKigqVREQAABQBQIANDTTIGI00BkZBkNGE0A0BkBkGQGRkaNAaAGQNBoGgDIAAYIGTI0DQAQAaGmmQMRpoDIyDIaMJoBoDIDIMgMjI0aA0AMgaDQNAGQAAwQMmRoGgAgA0NNMgYjTQGRkGQ0YTQDQGQGQZAZGRo0BoAZA0GgaAMgABggZMjQNABABoaaZAxGmgMjIMhowmgGgMgMgyAyMjRoDQAyBoNA0AZAADBAyZGgaAAmqU1NEgJqnptU/Sn4jRR6J6epk2pqb1Q/SgAPUGgyNNGjQ2SBpoAZAAGg0NB6mgDIAAAAA2oaApSREBNAARhGiYEaEwU8pvImlP0k2aam1GaGqbFNM1MHpTwmkepmyU9R6nqPKekHqNNPUxNGhp6n6p6QaZ6o9TG1GMqcoV9ly6nRanHlq6zPNbnGZNi6HSug+2nPiZ13XcnFYZW+45W11CumhzYhchOJ2GLLV1OBjBjGf4TptOddTSOcVxhqYZMYwZXZZY00zI1paX5X9J+b+f4e+x43RXSxXPOdquiGpduatGyXneN696M9t4HU2eR5XX/kPhP261NTx3JO1Ow7LyuDmeo9a7d351T1ZxnvnrvYnrXv/hXxPCeuYx2XsNmO003eg9J3Z6U7b23meJ4ri01OdzTk9BNO96brz+qT5nuvvH3ds/G+m/JcG/F2XYuhXlvO+jP7U3XgrzPN/lr8Sf1n6j4j7jZs+s/T0tNaNNYzTs12rxjwztHlnire3Nzc3N1wuBwOBwXBvZfoHpD7rFmR99V5vj3aXza3xdBbXMalubTg/jIv5dfAi54Pdc75j4z412n3Npj3Ld/ENm7a3b/Cod6h/ret1/5vn/C+l+gdslMvgPSLJ8d8q+U66fevYn/tW1chleEtNTGlcHCbLRlq0tHzF5tsbbZZfHjjLgZu42XCuC3NrdjTasZGNzgxPIrGqp7r3p7L2p5XjnpPSmTd5XtzqnB6U87zzg1Ol0zd0zsLszxR6lkxp35u6/teL0L0W922cR7Lu1lpL9CsHirzuM2T+BgsyViT6LHcm0/Vr6U/7LGGyJeqTEjt0PHWhF5mCT7R9mtlDwriYv0Tyr/OxYt6qp5r0mPVT0608TqnqMZaarU2nFwrTzzlrs1ed7z1ux60wyr4ydCaTi3enW8x68x0zU7tXSlcmPSW1mGpWJMg4zmPC2lK96tp0OE80y4MfEvnZj8zGluR6b22ki1Ou9V2nCd9xovcPvcYMZYy0lvN60ScZ45vN6yeCeeXFb1lVjnnCar5fwXwE2bzJ4HI1XVPXfXZMm44GUsMpYsmLB65TuVdm0cl0b+i/wGNN66XjeV7zuPpHcnK/juhhjdfId5jMdE5nN0dGmmm2zZs2cexD5n9p/dY352XsvXHaZNWWsmmS1atjR452nYudzvqv2HMRyvNNnlMcDl3R2+yx2uVrBubTW9icHDVtbNXlZm7jma1rM4VurZZd2y6nUau7ZXZ7bVU+mnoOVxZGMrVmvX60605JwmzGZhhhjTWtaaaMaaGTGmNMZasY0iX8VMUl8eepaIrzGSpemWOQyZORk2bNpjUybMmxqYmknCGCFynutfksaZpjTNMaaatM0xsxcGR0sociNqxNSmhhR1ZJPbsn8qyF0t2qH6iYBclclalbtTTcHTDsPaX6rlnElph2Jyumumtynv2Kk8GI7rsvXbIcJgHJOSaSXnnGaI3m87RtVXJOZ/YtgdTE6Wpha6ZlE8ayXkef1fh602r2WwvfMXtMdLlkfnLFdYYwYso+bWqm7yJqHXZGw2nrS5ZanSYnWlxBxMF1V940K2wdrI7R6OYf7DGGamMmTSbRhlS45xmVOumF1EyPCmHrrN8wwZOOrdNtLeMtzFzDlWnfTBxMk2NaXIZHBYxYLD4w8yju0ao65Vz1OIXoS9dLanwCe1PWrYuWMqf1if1z2k2yYfKJ741PDgno1ZQ8DRqvUny3mNoWTzGO6m1DkrJI8JiR5cSd+vZdGOO8nrMoc5+NDUFsMSXaZJeNlMmGLtJsovOsUp7I9S5VojKxF6bTVEelXqlfJobQr3LozSh2Jk7VcrVMfhXqszGWMzNqGhqZY0OadxkyyMssKugZR0KNFXBHlqwmJgTE/BNVMk6ItJXZMR0H47GpXv/DMOvNkmVuaV1PRfEdxuqc7Hcd+ZV/zTLaRxWk0nl9CdCeM6mn5rstHIBcpiuwmUZXeq81DacHI2rmrZ5SuE5mOZd6LQrZg9mx32TprA8BMo5jKN6yLTCi3WzQaZSuhzTtM1fUTGVpG8Tw+KXI0tjEpiWxtLYynOlktSbVlaI5kxP8TDH8kx50xoxi5KcA4pcja8KWLRlO/Ks6q06ergnvm1ca3Tq8Uw7LTUsmWyctXPWmpitl/uvGcWTGXGuAXDfhqazGmjkxcJW5hMMMMpYsXl2TZYtVOddG3XCarUt6Ptq9CZXSNzyuRzqRZOjsxdBbFVz6OA5HI43r1jityVlVpVkxmOsyaYWE1NTGq1sOVh36mHMcxtSvcy70edG0ZGR3I1Go1GRlV7mWWo1G0ZGRqlvH40l7o4m5xMWLLLYyNjnqc8556mdPqLJ31n/1nWOncxzG1tizrHs/Z+d2vP/B/l8wdJ6rHUn2nbbDq4p6htFtYzMMMTaZis1K5GKzGNmxhmUx2DDlZ/qNnIx41xnaMfCZWYaZWtNLTNW8ND4Fw1MyZOCdM428suKG1ehW8TesOydg7J+YYcD4cYR+8dFK6M4E3HM9ZfRNNL+Sn6rsl4DsrDl2HpPCnfxjGXtbZtYys1ttlyJ4T+BvexjGWRjMszK4Jpc77D3GyuVD7q0+G8m9G+2+rGm7cOR2y7FdtY2XUYx/oNlfRYxhMYyYZkyyg55enna9Kt/FFi6GMMwYwdwxWgxGMLKYmUyGExTKMZkMFhkymKuh0NOBNnBu+23LdwDoZYYzGGMxtORaTU1pjTGWTTGGtMrNWUsyyTTLLG1qy2ZjbK2DBllWqxMtBMaYZQmcE7zvvRcTkclUwdkxTaSdyySt/7fpL+T1v516Ji97fwr5JbLu305zMn5+GMTTZ9F+y7ExwmGVfG44yxn3dLv6l5i+Wth1jCrDq21nW9LqvvDzz3Vf3LLH/O/32TJ/erx3bXftO4eF+G956D952K/An4NfvOpjFjExjevP/UmE0fIoZXx6/w6lX/no3D0bLt+ixjieBM6ksRd0yB4Lt2SwYNE+gd1detlZWUnpiZfGfFaK+4PyCa/v18V8X75pe9fLXzp7l3VjF76vWZmHwGz1IZNWT7b8yddJ4q5kyrVdfru6atWc7bVYztL9Jf4GXvT+Y8m9/YsXP6H018a8D4XVOqvfzqeR+6yZOD8dPv0+U7/q5Pl+2dNb0MjzGVH5p6MNQ7cOWvw62U9aHE8DprDek+McLyvDz+te+9Zhq5+YTruufMcWMabqysTmZVWjKPfnK0wyVcrsuhjZRdLkHNvD72b9abriOSGIxiLixMOoalNPXzy+wT/tf+U6HHONfsz+xe8ufHBdQWWGWLA9if0rsnmrxK5LvRZQeWsTCsrmOYy8VteVfuRfcVTtDLItLIsMYxZLdU/DbtSemxF6Z6Zo5WBXE4tFdCyVMMXMTEMZXVlS6Xec2T4e0tHsRcEuWshcJ2YsNF5rUx1E8ifCq6Z+ZP7qdCeu/aTwFd53l16/o0NOw6O3dLavP4Hbi4RdmuDk6DoYaninC0+o4uZjbJ7Rxeu0/FbuFg+q7DVS6fQe0rZ6NDGUNNU6DEqOaLTicKnYZMnBWruljQxoaS3dZhocDge0bSTyOvdAbG5hxe2xji7E/L55xX13wWNDi6HCekcFxfCPGxY0MXC+s7afWaMdDyjyr+o8Rudm/NabOZvdl274zH4f5XK9z6On1Pe/K5TdPAslg77BjuO6Y3eO7GqvOPG/stknp1leyvLL0Z7bl9I4noMvLkzytLhWYzrOZzLXCORe028rORzOg4N/L0HlMOQ3Pgmnbb6KczlabORpu980q37TBqRu0/p3PO6234Bl03Ynuz+9W7gnsEcmvYaYY3aMYY0wx3pYd+ujsXauWdaY5Xkbtl23fPzFHiDB/QMo0yFjBllYxTQYYyxkrwn7JufwJ/PfgJ+C83X69ni6zvXcnyXabv0ncbLwsceS+RNlyN2mnneJtX0ngYO0+e+0+UnA+Wch3ji8hj5an4h+i6XBySU4n+R0roVcbw5yvHrmr4Yw8Y7x6c+9POPYHI5HI5HI5HI5HGXGww4nE4nrVyOR8XeqPEO7PLOiukYa3Novk5hV4cdtYZLI93e+uxff2jRo0aNGjRo0aNG1bVtW1dy3m83m8+tQ5ZzHw3nObwOu8La9Rc1dtkdS8A3eTk823tnktXWlxN6Oixe06zrN70Isd9jiOgZFq9yfkPqP/SLhN2Myl8jDM43bl1nbcb4cO57jlh8Jow6pzXZdL4dyODTuuhu77FyO27DdwdRxmvO+O+3N2+BdqyTwLHVczDVY4UPE4O66/ZO2cx1LFzVdSXtF7G4HMbrauOHRw6c8FdZ5m9fHZHYZXfTlZquyynSyTTKke6vcffSD9pzPA/G7n7jxPmuhc1DHMynPMrGL6AdewYmwu5ko+UUyTwrMv27rPH1v1nGqd87+p6N6LU8k3NEng53xXyHS97+44OSg/sy/hn+Se6yfYNjW0/uTgP+PvWYzLMmjhcLB/gGpri6H83/84eUXWT6T9Hsv7785z/7z4icpW+zfXypuR7rx/gMdZb1/wC678pcs8/2a3mDitGHxl9mfPlll5MafWWqxk/eYuTDgcNMzDGWLWvsuglNxs53GtN6uWpktlW1tZZYcuinMMWmnNnJydze3b2Y1McBxrBkXw799izLMZZYyy0TkbsGM4p03S2uVu5s/XXUdSdec6smVxZYYGpVmT8A+8ajuEyV5FatkvVru2x6uxGXXbH4A+jvgP4GMYy3iPLXzq/6z65+E005ey+cwMZD3fZcqc6xpjTFjQ0P3U+e++cPYmTIwj0nrK5NPTfl3WvpfLtXDcb2HQMudYOxFXQBor4L4T6vrOauFctYXJQ++NUWmJe5bmx1jDiZS1dTqWxo4GR8jm3fttpmPHppk9PEyv4/y8/sO07XacOmcqc0x2Vi9BvNJvN5oW8x4mOsydpidRxMYJPx06m1bqPzq9KtK8sxXNXFodD/+MYYaJTLwOhc9brCsV18oOR1i4tXChyTkq4lf4y1Ke+9axjDHqs1mfBbMXuP4Hzi+X7t8vzv7bHerrUPgPCxhjre4fXdfLNtNM+Jd+Zdh8xd8wP87uNPoPgv4W7/5P2BuxfsMabNnMnza+54Pdi5U671GPZY8CehX8Voeoo7FHpkeEc6715FwHZrIrUrHaviPUbPZHND+IhczrP6FcYvhOZ0Di/ETt0OI+YwNWR9r7tpf6WDeZKZDB1+z2IthOl1mPyb5FluvEx9h9d0NnM0Y1XPFkWIsk1WotJ0PBMmkvjvQTd0e71tfeV+8r8lQ/tpzpsmxJ+InrI/dj2UajUajVTUajatRqNRtGo1Go1Go4wjeMpZFMVV9CHbofPraLsJ3JpWV2XOoanCuFky4y3PPNxucK2uKC1Lbdb1eo+m5XomN6HfeZsabHLHRX/K+offtNGGmHWctcVcG44MdSqsOLY9VzX+Zxfxn2HPdWTpzWvkrtJ8M5zorrKcquRytJ5N5DZmcaW02l76nWO+BqPXm1A2Ry/0q71dH/mqrqeFjkYxjEXtsX8qubTk67rGycyqsdm4tZx5D6D5hhi0waaWmiaMP81Yjii5qxPlPuU/GfTL1Y5E6Jyfiq63qTa39A4J0sOGDgO9WF9bOXl0XfPRbsY2bPNKPy1YrFYrFYmRhhlTIyMjJWJYZHXuCXI8OoXsvfljGLFicNifpp2XunoPiG1wtx3p1Tah+/DD66OnVtVXP9rKbVxOnL0tR/rHtqB5UDErUVcl11D4qqvjpOcxX7armUNJB3LpW6bxVvD08e8h3odKKvyCFZBdSh2FVcST9xV3n3T8t1j7Kr9qgrqXg+13Pt5U7JCvFXVIV1YG5lRhkVYZJYYDDD4KOIMoHCp26WS8GB7uBh2zIdgq/PKyInjV2STShuoapUdCpX1yTwqq/z1VvET7Kh5nVPkO8YyxjLt2MaaMmWTLQvx3qnzltnXW0p2jxgbEtSny/Osv8Y9pLMXYoHVPAhkVdWVeODhR6q9/Sxe2liwwZWMVvFXfRkeIDxAePUPIrdJ4ey6yquzH+PD/bUOWAu05qVHtFd8rrKHSoeNIOUqrYr3FXyToqfYJgwmJdKpXXOwYYegNNGMzfZPp/t3t/DVs4zjNTN61rRqaWaa4NYbRjTa0tWwy2Y2tGN8ZO8ofNKq4j9SL7I+cSm4/6ovLV5HNXLI0jJidwrtk6ynCaP6Z++GjRlWS3tLeW129Mi9evxU9mtz6s5J3Z7M2ngTgnKvmpomxpaLCzPfmx0JWE+m3NLDDGOX47RctdYYNK5jakdqLkRlI39n590T5zctGSwwZZDJj6kW8XSi6ot2MmWWJ0DUT3nuvebBudScjZ79g8cWJ8av0k+/bE5WKd5MdbFpbDVMxu1DVMmtNZGJvq1mtRbn6M+g/kP0FwDwr7quZs7xosNGpbscyxhhd9TyJyFwbLcxlTasg75vW7TsV5K7ji44XPMMrdoj+Y3rT0Hie62nlYV/pwczzOmdLqLhYkzGMzCZWGMQzGMSsZYY6Di1t4nlJ+Em63mJxrVLxPbYxNEdgc1dU2iOKyoYYWjNrEeHTYybVk0atSa7ehuwsWMWTqn1TrnS6hYsi71d1+s+k+ic70e20fzE/VaTdxT9ZtU4GIXdeNx3X77guYYfpHeTQjaMX6brOu4OY4K7Y2d9mbHarI5ox3p4GpJ2Vd/Tst60f7j999pppjR+Q/Qf8J/VaORs3cji7FfFuN61+ui9s8hix1OCh5KGVV23BPXvZfz3CLyHpix+exi8z/KnCnosY2eunor+cxyPO/xJ0vKey9OvE9VjqaYu0x3Z3jd6o2b1T12D+F8l232lwaaacD5LE8LBxu7WTlbWraWpew8Xexjel3E+wWD4APITdNqR8F3R3T0lunCQ4GaE9R37DxeCYfcHi4xci5ovKfxVs55y2hf+65E/Xdp6jR5nrebTmi5incpkyOjs50JvrZwstbbW6kfuuQw+2mykf/EXNFzxfKTrxew929TR6bWnGL//F3JFOFCQT3K4lQ"

    kernels = Kernel(
        bz2.decompress(base64.b64decode(quantization_code)),
        [
            "int4WeightCompression",
            "int4WeightExtractionFloat",
            "int4WeightExtractionHalf",
            "int8WeightExtractionFloat",
            "int8WeightExtractionHalf",
        ],
    )
except Exception as exception:
    kernels = None


class W8A16Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, quant_w: torch.Tensor, scale_w: torch.Tensor, weight_bit_width):
        ctx.inp_shape = inp.size()
        ctx.weight_bit_width = weight_bit_width
        out_features = quant_w.size(0)
        inp = inp.contiguous().view(-1, inp.size(-1))
        weight = extract_weight_to_half(quant_w, scale_w, weight_bit_width)
        ctx.weight_shape = weight.size()
        output = inp.mm(weight.t())
        ctx.save_for_backward(inp, quant_w, scale_w)
        return output.view(*(ctx.inp_shape[:-1] + (out_features,)))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inp, quant_w, scale_w = ctx.saved_tensors
        weight = extract_weight_to_half(quant_w, scale_w, ctx.weight_bit_width)
        grad_output = grad_output.contiguous().view(-1, weight.size(0))
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(inp)
        return grad_input.view(ctx.inp_shape), grad_weight.view(ctx.weight_shape), None, None


def compress_int4_weight(weight: torch.Tensor):  # (n, m)
    with torch.cuda.device(weight.device):
        n, m = weight.size(0), weight.size(1)
        assert m % 2 == 0
        m = m // 2
        out = torch.empty(n, m, dtype=torch.int8, device="cuda")
        stream = torch.cuda.current_stream()

        gridDim = (n, 1, 1)
        blockDim = (min(round_up(m, 32), 1024), 1, 1)

        kernels.int4WeightCompression(
            gridDim,
            blockDim,
            0,
            stream,
            [ctypes.c_void_p(weight.data_ptr()), ctypes.c_void_p(out.data_ptr()), ctypes.c_int32(n), ctypes.c_int32(m)],
        )
        return out


def extract_weight_to_half(weight: torch.Tensor, scale_list: torch.Tensor, source_bit_width: int):
    if source_bit_width == 8:
        func = kernels.int8WeightExtractionHalf
    elif source_bit_width == 4:
        func = kernels.int4WeightExtractionHalf
    else:
        assert False, "Unsupported bit-width"

    with torch.cuda.device(weight.device):
        n, m = weight.size(0), weight.size(1)
        out = torch.empty(n, m * (8 // source_bit_width), dtype=torch.half, device="cuda")
        stream = torch.cuda.current_stream()

        gridDim = (n, 1, 1)
        blockDim = (min(round_up(m, 32), 1024), 1, 1)

        func(
            gridDim,
            blockDim,
            0,
            stream,
            [
                ctypes.c_void_p(weight.data_ptr()),
                ctypes.c_void_p(scale_list.data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_int32(n),
                ctypes.c_int32(m),
            ],
        )
        return out


class CacheTensor():
    def __init__(self, *args, **kwargs):
        self.tensor = torch.empty(*args, **kwargs)

    def to(self, *args, **kwargs):
        self.tensor = self.tensor.to(*args, **kwargs)

    def data_ptr(self):
        return self.tensor.data_ptr()


class QuantizedLinear(Linear):
    def __init__(self, weight_bit_width: int, weight_tensor=None, bias_tensor=None, quantized_weight=None,
                 quantized_weight_scale=None, quantization_cache=None, empty_init=False, *args, **kwargs):
        super(QuantizedLinear, self).__init__(*args, **kwargs)
        self.weight_bit_width = weight_bit_width
        self.quantization_cache = quantization_cache

        if (quantized_weight is not None) and (quantized_weight_scale is not None):
            del self.weight
            self.weight = Parameter(quantized_weight.to(kwargs["device"]), requires_grad=False)
            self.weight_scale = Parameter(quantized_weight_scale.to(kwargs["device"]), requires_grad=False)
        else:
            shape = self.weight.shape
            del self.weight

            if weight_tensor is None or empty_init:
                self.weight = torch.empty(
                    shape[0], shape[1] * weight_bit_width // 8, dtype=torch.int8, device=kwargs["device"]
                )
                self.weight_scale = torch.empty(shape[0], dtype=kwargs["dtype"], device=kwargs["device"])
            else:
                self.weight_scale = (weight_tensor.abs().max(dim=-1).values / ((2 ** (weight_bit_width - 1)) - 1)).to(kwargs["dtype"])
                self.weight = torch.round(weight_tensor / self.weight_scale[:, None]).to(torch.int8)
                if weight_bit_width == 4:
                    self.weight = compress_int4_weight(self.weight)

            self.weight = Parameter(self.weight.to(kwargs["device"]), requires_grad=False)
            self.weight_scale = Parameter(self.weight_scale.to(kwargs["device"]), requires_grad=False)

        if bias_tensor is not None:
            self.bias = Parameter(bias_tensor.to(kwargs["device"]), requires_grad=False)
        else:
            self.bias = None

    def reset_parameters(self):
        """To accelerate initialization"""
        pass

    def forward(self, input):
        output = W8A16Linear.apply(input, self.weight, self.weight_scale, self.weight_bit_width)
        if self.bias is not None:
            output = output + self.bias
        return output

    def _apply(self, fn):
        self_obj = super()._apply(fn)
        if self.quantization_cache is not None:
            self.quantization_cache.to(self_obj.weight.device)
            self.quantization_cache.to(self_obj.weight_scale.dtype)
        return self_obj
    

class QuantizedEmbedding(Embedding):  # TODO: backward, check empty_init
    def __init__(self, weight_bit_width: int, weight_tensor=None, quantized_weight=None, quantized_weight_scale=None,
                 empty_init=False, *args, **kwargs):
        super(QuantizedEmbedding, self).__init__(*args, **kwargs)
        self.weight_bit_width = weight_bit_width

        if (quantized_weight is not None) and (quantized_weight_scale is not None):
            del self.weight
            self.weight = Parameter(quantized_weight.to(kwargs["device"]), requires_grad=False)
            self.weight_scale = Parameter(quantized_weight_scale.to(kwargs["device"]), requires_grad=False)
        else:
            shape = self.weight.shape
            del self.weight

            if weight_tensor is None or empty_init:
                self.weight = torch.empty(
                    shape[0], shape[1] * weight_bit_width // 8, dtype=torch.int8, device=kwargs["device"]
                )
                self.weight_scale = torch.empty(shape[0], dtype=kwargs["dtype"], device=kwargs["device"])
            else:
                self.weight_scale = (weight_tensor.abs().max(dim=-1).values / ((2 ** (weight_bit_width - 1)) - 1)).to(
                    kwargs["dtype"])
                self.weight = torch.round(weight_tensor / self.weight_scale[:, None]).to(torch.int8)
                if weight_bit_width == 4:
                    self.weight = compress_int4_weight(self.weight)

            self.weight = Parameter(self.weight.to(kwargs["device"]), requires_grad=False)
            self.weight_scale = Parameter(self.weight_scale.to(kwargs["device"]), requires_grad=False)

    def forward(self, input):
        original_weight = extract_weight_to_half(weight=self.weight, scale_list=self.weight_scale,
                                                    source_bit_width=self.weight_bit_width)
        output = F.embedding(
            input, original_weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse
        )
        return output


class CpmKernelQuantizer(QuantizerBase):
    def _process_model_before_weight_loading(self, model:nn.Module, **kwargs):
        """从chagglm-6b移植过来的的量化，方便以int8和int4进行推理
        源链接：https://huggingface.co/THUDM/chatglm-6b/blob/main/quantization.py
        
        Replace fp16 linear with quantized linear
        这里修改了hard code, 可以适配其他模型
        target_modules: str/list, 指定对某些层做量化
        """
        quantization_config = kwargs['quantization_config']
        quantization_bit:int = quantization_config['quantization_bit']
        use_quantization_cache:bool = quantization_config.get('use_quantization_cache', False)
        empty_init:bool = quantization_config.get('empty_init', False)
        target_modules:Union[str, List] = quantization_config.get('target_modules', None)

        if not is_package_available('cpm_kernels'):
            raise ModuleNotFoundError('Module `cpm_kernels` not found, you may use `pip install cpm_kernels`')

        # 把meta权重to_empty(device='cpu'), 执行后就不是meta了
        if has_meta_param(model):
            model.apply(model.init_meta_weights)

        model.half()  # 确保模型是fp16
        modules_trans = {}
        for name, module in model.named_modules():
            # target_modules=None, 表示对所有Linear层替换
            if (target_modules is None) and isinstance(module, Linear):
                modules_trans[name] = module
            elif (target_modules is not None) and isinstance(module, Linear):
                if isinstance(target_modules, str):
                    target_module_found = re.fullmatch(target_modules, name)
                else:
                    target_module_found = any(name.endswith(target_key) for target_key in target_modules)
                if target_module_found:
                    modules_trans[name] = module
        
        # TODO: 暂时只支持cuda
        current_device = torch.cuda.current_device()
        dtype = torch.half

        QuantizedLinearWithPara = partial(
            QuantizedLinear,
            weight_bit_width=quantization_bit,
            bias=True,
            dtype=dtype,
            empty_init=empty_init
        )
            
        cache = dict()
        for name, module in tqdm(modules_trans.items(), desc='Quantize linear layers'):
            cache_name = re.sub(r'\.[0-9]+\.', '.', name)
            if use_quantization_cache and (cache_name not in cache):
                n, m = module.weight.size(0), module.weight.size(1)
                cache[cache_name] = CacheTensor(n, m, dtype=dtype, device=current_device, requires_grad=False)

            module_quant = QuantizedLinearWithPara(
                    weight_tensor=module.weight.to(current_device),
                    bias_tensor=module.bias,
                    in_features=module.in_features,
                    out_features=module.out_features,
                    device=module.weight.device,
                    quantization_cache=cache.get(cache_name)
                )
            del module
            # 赋值
            name_new = list(name)
            for iter_ in re.finditer(r'\.[0-9]+\.', name):
                iter_str = name[iter_.start():iter_.end()]
                name_new[iter_.start():iter_.end()] = [''] * (iter_.end()-iter_.start())
                name_new[iter_.start()] = '[' + iter_str[1:-1] + '].'
            exec('model.' + ''.join(name_new) + ' = module_quant')
        return model

    def _process_model_after_weight_loading(self, model, **kwargs):
        return model

    def is_serializable(self, *args, **kwargs):
        return True

    @property
    def is_trainable(self, *args, **kwargs):
        return False

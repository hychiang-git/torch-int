import torch
from .._CUDA import linear_relu_a8_w8_b8_o8
from .._CUDA import linear_with_token_scaling
from ..functional.quantization import (
    quantize_per_tensor_absmax,
    quantize_weight_per_channel_absmax,
    fake_quantize_activation_per_tensor_absmax,
    fake_quantize_activation_per_token_absmax,
)


class W8A8B8O8LinearReLU(torch.nn.Module):
    # For fc1
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight', torch.randint(-127, 127, (self.out_features,
                                                                 self.in_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer('bias', torch.zeros(
            (1, self.out_features), dtype=torch.int8, requires_grad=False))
        self.register_buffer('a', torch.tensor(alpha))
        self.register_buffer('b', torch.tensor(beta))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        y = linear_relu_a8_w8_b8_o8(x, self.weight, self.bias,
                                    self.a, self.b)
                                    # self.a.item(), self.b.item())
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale, output_scale):
        # TODO: add zero-point to prevent the bit waste
        int8_module = W8A8B8O8LinearReLU(
            module.in_features, module.out_features)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        int8_bias, bias_scale = quantize_per_tensor_absmax(module.bias)
        alpha = input_scale * weight_scale / output_scale
        beta = bias_scale / output_scale
        int8_module.weight = int8_weight
        int8_module.bias = int8_bias
        int8_module.a = alpha
        int8_module.b = beta
        return int8_module



# class W8A8B32O32Linear(torch.nn.Module):
#     # For fc2 and out_proj
#     def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features

#         self.register_buffer('weight', torch.randint(-127, 127, (self.out_features,
#                                                                  self.in_features), dtype=torch.int8, requires_grad=False))
#         self.register_buffer('bias', torch.zeros(
#             (1, self.out_features), dtype=torch.int32, requires_grad=False))
#         self.register_buffer('a', torch.tensor(alpha))
#         self.register_buffer('b', torch.tensor(beta))

#     def to(self, *args, **kwargs):
#         super().to(*args, **kwargs)
#         self.weight = self.weight.to(*args, **kwargs)
#         self.bias = self.bias.to(*args, **kwargs)
#         return self

#     @torch.no_grad()
#     def forward(self, x):
#         x_shape = x.shape
#         x = x.view(-1, x_shape[-1])
#         y = linear_a8_w8_b32_o32_with_scaling(
#             x, self.weight, self.bias, self.a.item(), self.b.item())
#         y = y.view(*x_shape[:-1], -1)
#         return y

#     @staticmethod
#     def from_float(module: torch.nn.Linear, input_scale, output_scale):
#         int8_module = W8A8B32O32Linear(
#             module.in_features, module.out_features)
#         int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
#         module.bias = module.bias.float()
#         bias_scale = module.bias.abs().max() / (2**31 - 1)
#         int32_bias = (module.bias / bias_scale).round().to(torch.int32)
#         alpha = input_scale * weight_scale / output_scale
#         beta = bias_scale / output_scale
#         int8_module.weight = int8_weight
#         int8_module.bias = int32_bias
#         int8_module.a = alpha
#         int8_module.b = beta
#         int8_module.input_scale = input_scale
#         int8_module.output_scale = output_scale
#         int8_module.weight_scale = weight_scale
#         int8_module.bias_scale = bias_scale
#         return int8_module


if __name__ == "__main__":
    import torch.nn.functional as F
    import torch.utils.benchmark as benchmark

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    device = torch.device("cuda:0")
    # x = torch.arange(4).reshape((1, 4)).to(torch.float32).cuda()
    # w = torch.arange(32).reshape((8, 4)).to(torch.float32).cuda()
    # b = torch.zeros((1, 8)).to(torch.float32).cuda()
    # alpha = torch.ones((1, 1)).to(torch.float32).cuda()
    
    L = 1
    # in_feat = 2560
    # out_feat = 2560*4
    in_feat = 768
    out_feat = 768*4
    x = torch.rand((L, in_feat)).to(torch.float16).cuda()
    w = torch.rand((out_feat, in_feat)).to(torch.float16).cuda()
    b = torch.zeros((out_feat)).to(torch.float16).cuda()
    alpha = torch.rand((L, 1)).to(torch.float16).cuda()

    # print(x)
    # print(w)
    y = alpha*F.linear(x, w, b)
    print(y.shape)
    print(y)

    print()

    # print(x)
    # print(w)
    y_ = linear_with_token_scaling(x, w, b, alpha)
    print(y_.shape)
    print(y_)

    assert torch.allclose(y, y_, rtol=1e-02, atol=1e-02)





    # x_scale = x.abs().amax() / 127
    # qx = (x / x_scale).round().to(torch.int8)

    # w_scale = w.abs().amax() / 127
    # qw = (w / w_scale).round().to(torch.int8)

    # b_scale = b.abs().amax() / 127
    # qb = (b / b_scale).round().to(torch.int8)

    # q_y = linear_relu_a8_w8_b8_o8(qx, qw.t(), qb, 1.0, 1.0)
    # print(qx.shape, qw.shape, qw.t().shape, q_y.shape, q_y.dtype)
    # t0 = benchmark.Timer(
    #     stmt='linear_relu_a8_w8_b8_o8(qx, qw, qb, 1.0, 1.0)',
    #     setup='from __main__ import linear_relu_a8_w8_b8_o8',
    #     globals={'qx': qx, 'qw': qw.t().contiguous(), 'qb': qb})
    # print(t0.timeit(1000))

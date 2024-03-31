#ifndef LINEAR_H
#define LINEAR_H
#include <torch/types.h>

// used by fc1, return INT8
torch::Tensor linear_relu_a8_w8_b8_o8(
    torch::Tensor input,  // INT8
    torch::Tensor weight, // INT8
    torch::Tensor bias,   // INT8
    float alpha,          // FP32
    float beta            // FP32
);
// used by out_proj and fc2, return INT32
torch::Tensor linear_with_token_scaling(
    torch::Tensor input,  // FP32
    torch::Tensor weight, // FP32
    torch::Tensor bias,   // FP32
    torch::Tensor alpha  // FP32
);

#endif // LINEAR_HS
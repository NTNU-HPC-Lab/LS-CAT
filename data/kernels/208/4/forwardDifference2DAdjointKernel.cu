#include "includes.h"
__global__ void forwardDifference2DAdjointKernel(const int cols, const int rows, const float* dx, const float* dy, float* target) {
for (auto idy = blockIdx.y * blockDim.y + threadIdx.y + 1; idy < cols - 1;
idy += blockDim.y * gridDim.y) {
for (auto idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
idx < rows - 1; idx += blockDim.x * gridDim.x) {
const auto index = idx + rows * idy;

target[index] =
-dx[index] + dx[index - 1] - dy[index] + dy[index - rows];
}
}
}
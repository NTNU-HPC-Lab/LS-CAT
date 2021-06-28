#include "includes.h"
__global__ void forwardDifference2DKernel(const int cols, const int rows, const float* data, float* dx, float* dy) {
for (auto idy = blockIdx.y * blockDim.y + threadIdx.y + 1; idy < cols - 1;
idy += blockDim.y * gridDim.y) {
for (auto idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
idx < rows - 1; idx += blockDim.x * gridDim.x) {
const auto index = idx + rows * idy;

dx[index] = data[index + 1] - data[index];
dy[index] = data[index + rows] - data[index];
}
}
}
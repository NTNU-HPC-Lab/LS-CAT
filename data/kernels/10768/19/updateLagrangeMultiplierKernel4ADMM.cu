#include "includes.h"
__global__ void updateLagrangeMultiplierKernel4ADMM(float* u, float* v, float* lam, float* temp, float mu, uint32_t w, uint32_t h, uint32_t nc) {
uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;
uint32_t c = threadIdx.z + blockDim.z * blockIdx.z;

if(x < w && y < h && c < nc) {
uint32_t index = x + w * y + w * h * c;
temp[index] = u[index] - v[index];
lam[index] = lam[index] + temp[index] * mu;
}
}
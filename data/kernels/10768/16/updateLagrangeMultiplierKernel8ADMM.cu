#include "includes.h"
__global__ void updateLagrangeMultiplierKernel8ADMM(float* u, float* v, float* w_, float* z, float* lam1, float* lam2, float* lam3, float* lam4, float* lam5, float* lam6, float* temp, float mu, uint32_t w, uint32_t h, uint32_t nc) {
uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;
uint32_t c = threadIdx.z + blockDim.z * blockIdx.z;

if(x < w && y < h && c < nc) {
uint32_t index = x + w * y + w * h * c;
temp[index] = u[index] - v[index];
lam1[index] = lam1[index] + mu * (u[index] - v[index]);
lam2[index] = lam2[index] + mu * (u[index] - w_[index]);
lam3[index] = lam3[index] + mu * (u[index] - z[index]);
lam4[index] = lam4[index] + mu * (v[index] - w_[index]);
lam5[index] = lam5[index] + mu * (v[index] - z[index]);
lam6[index] = lam6[index] + mu * (w_[index] - z[index]);
}
}
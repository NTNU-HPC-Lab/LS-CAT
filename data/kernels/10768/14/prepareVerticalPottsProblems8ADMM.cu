#include "includes.h"
__global__ void prepareVerticalPottsProblems8ADMM(float* in, float* u, float* v, float* w_, float* z, float* weights, float* weightsPrime, float* lam1, float* lam4, float* lam5, float mu, uint32_t w, uint32_t h, uint32_t nc) {
uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;
uint32_t c = threadIdx.z + blockDim.z * blockIdx.z;

if(x < w && y < h && c < nc) {
uint32_t index = x + w * y + w * h * c;
uint32_t weightsIndex = x + w * y;

v[index] = (weights[weightsIndex] * in[index] + 2 * mu * (u[index] + w_[index] + z[index])
+ 2 * (lam1[index] - lam4[index] - lam5[index])) / weightsPrime[weightsIndex];

}
}
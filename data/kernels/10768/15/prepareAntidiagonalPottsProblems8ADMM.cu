#include "includes.h"
__global__ void prepareAntidiagonalPottsProblems8ADMM(float* in, float* u, float* v, float* w_, float* z, float* weights, float* weightsPrime, float* lam3, float* lam5, float* lam6, float mu, uint32_t w, uint32_t h, uint32_t nc) {
uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;
uint32_t c = threadIdx.z + blockDim.z * blockIdx.z;

if(x < w && y < h && c < nc) {
uint32_t index = x + w * y + w * h * c;
uint32_t weightsIndex = x + w * y;

z[index] = (weights[weightsIndex] * in[index] + 2 * mu * (u[index] + v[index] + w_[index])
+ 2 * (lam3[index] + lam5[index] + lam6[index])) / weightsPrime[weightsIndex];

}
}
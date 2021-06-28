#include "includes.h"
// filename: eeTanh.cu
// a simple CUDA kernel to square the elements of a matrix



extern "C"   // ensure function name to be exactly "eeTanh"
{





















}
__global__ void noactivationGradient(int N, int M, float *z, float *tanh_grad_z, int seed, float D) {

int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;

int index = j*N + i;

float scaleFactor = __fdividef(1.0, __fsub_rn(1.0, D));

if (i < N && j < M) {
curandState_t state;
curand_init( (seed << 20) + index, 0, 0, &state);

float F = curand_uniform(&state);
// float F = 0.5;

if (D != 0.0) {
if (F < D) {
z[index] = 0.0;
tanh_grad_z[index] = 0.0;
}
else {
tanh_grad_z[index] = scaleFactor;
z[index] = __fmul_rn(scaleFactor, z[index]);
}
}
else {
tanh_grad_z[index] = 1.0;
}
}
}
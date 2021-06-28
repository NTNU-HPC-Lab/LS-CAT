#include "includes.h"
__global__ void kernExp(double* A, double* bias) {
int b = blockIdx.y * gridDim.x + blockIdx.x;
int i = b * blockDim.x + threadIdx.x;
A[i] = exp(A[i] - *bias);
}
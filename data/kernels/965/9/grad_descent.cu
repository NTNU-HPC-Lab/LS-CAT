#include "includes.h"
__global__ void grad_descent(float *odata, const float *idata, int size) {
int t = blockIdx.x * blockDim.x + threadIdx.x;
if (t < size) {
odata[t] -= LEARNIG_RATE * idata[t];
}
}
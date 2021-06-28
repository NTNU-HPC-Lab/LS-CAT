#include "includes.h"
extern "C" {
}

#define IDX2C(i, j, ld) ((j)*(ld)+(i))
#define SQR(x)      ((x)*(x))                        // x^2

__global__ void multiply_arrays(double* signals, double const* weights){
signals[blockIdx.x * blockDim.x + threadIdx.x] *= weights[blockIdx.x * blockDim.x + threadIdx.x];
}
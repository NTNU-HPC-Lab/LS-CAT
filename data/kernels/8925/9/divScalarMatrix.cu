#include "includes.h"
__global__ void divScalarMatrix(double *dMatrix, double *dScalar, int dSize){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
while (tid < dSize) {
dMatrix[tid] = dMatrix[tid]/dScalar[0];
tid  += blockDim.x * gridDim.x;
}
}
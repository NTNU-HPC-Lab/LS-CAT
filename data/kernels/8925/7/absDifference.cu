#include "includes.h"
__global__ void absDifference(double *dDifference, double *dSup, double *dLow, int dSize){
int tid = threadIdx.x + blockIdx.x * blockDim.x;

while (tid < dSize) {
double a = dSup[tid];
double b = dLow[tid];
dDifference[tid] = (a > b) ? (a - b) : (b - a);
tid  += blockDim.x * gridDim.x;
}
}
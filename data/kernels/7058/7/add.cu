#include "includes.h"
__global__ void add(int * A, int * B, int * C){
int thread = blockIdx.x*blockDim.x + threadIdx.x;
C[thread] = A[thread] + B[thread];
}
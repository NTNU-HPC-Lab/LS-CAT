#include "includes.h"
__global__ void dot(float * val, int *row_ind, int *col_ind, int nnz, float * ret, float * w) {
const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
if (tid < nnz) {
int r = row_ind[tid];
int c = col_ind[tid];
float v = val[tid];
atomicAdd(&ret[r], v * w[c]);
}
}
#include "includes.h"
__global__ void grad(float * val, int * row_ind, int *col_ind, float * mat_err, int nnz, float *act, float *label, float *w, float learning_rate) {
const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
if (tid < nnz) {
int r = row_ind[tid];
int c = col_ind[tid];
float v = val[tid];
mat_err[tid] = abs(label[r] - act[r]);
float err = v * (label[r] - act[r]);
atomicAdd(&w[c], learning_rate * err);
}
}
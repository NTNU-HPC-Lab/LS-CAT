#include "includes.h"
extern "C" {
}

const double TOLERANCE = 1.0e-10;

/*
cgsolver with CUDA support solves the linear equation A*x = b where A is of size m x n
*/

__global__ void mvm_gpu(double *A_cuda, double *X_cuda, double *Y_cuda, int *m_locals_cuda, int *A_all_pos_cuda, int n, int nthreads){
int t = blockIdx.x * blockDim.x + threadIdx.x;

if (t < nthreads){
for (int i=A_all_pos_cuda[t]; i<A_all_pos_cuda[t]+m_locals_cuda[t]; ++i) {
Y_cuda[i] = 0.;
for (int j=0; j<n; ++j)
Y_cuda[i] += A_cuda[i * n + j] * X_cuda[j];
}
}
}
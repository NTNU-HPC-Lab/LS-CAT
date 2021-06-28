#include "includes.h"
/**
* C file for parallel QR factorization program usign CUDA
* See header for more infos.
*
* 2016 Marco Tieghi - marco01.tieghi@student.unife.it
*
*/



#define THREADS_PER_BLOCK 512   //I'll use 512 threads for each block (as required in the assignment)

__global__ void r1_update(double *A, int m, int n, int lda, double *col, int ldc, double *row) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int idy = blockIdx.y * blockDim.y + threadIdx.y;

//A(:,ii+1:n−1)=A(:,ii+1:n−1)−A(:,ii)*R(ii,ii+1:n−1)
if (idx < m && idy < m) {
for (int ii=0; ii < n-1; ii++) {
A[idx*lda + ii+1] = A[idx*lda + ii+1] - col[idy*ldc] * row[ii+1];
}
}
}
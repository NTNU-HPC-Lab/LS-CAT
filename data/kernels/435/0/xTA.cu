#include "includes.h"
/**
* C file for parallel QR factorization program usign CUDA
* See header for more infos.
*
* 2016 Marco Tieghi - marco01.tieghi@student.unife.it
*
*/



#define THREADS_PER_BLOCK 512   //I'll use 512 threads for each block (as required in the assignment)

__global__ void xTA (double *y, int k, double*A, int m, int lda, double *x, int ldx) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
double s;   //It memorizes the sum

if (idx < k) {
for (int ii = 0; ii < m; ii++) {    //Moving through rows
s += x[ii * ldx] * A[idx + ii*lda];
}
y[idx] = s;  //Adding the sum to result vector
}
}
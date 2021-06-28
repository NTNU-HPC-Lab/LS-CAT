#include "includes.h"
__global__ void chol_kernel_cudaUFMG_zero(float * U, int elem_per_thr) {
// Get a thread identifier
int tx = blockIdx.x * blockDim.x + threadIdx.x;
int ty = blockIdx.y * blockDim.y + threadIdx.y;

int tn = ty * blockDim.x * gridDim.x + tx;

for(unsigned i=0;i<elem_per_thr;i++){
int iel = tn * elem_per_thr + i;
int xval = iel % MATRIX_SIZE;
int yval = iel / MATRIX_SIZE;

if(xval == yval){
continue;
}

// if on the upper diagonal...
if(yval < xval){
xval = MATRIX_SIZE - xval - 1;
yval = MATRIX_SIZE - yval - 1;
}
int iU = xval + yval * MATRIX_SIZE;
U[iU] = 0;
}

}
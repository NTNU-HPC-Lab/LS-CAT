#include "includes.h"
__global__ void __initSeq(int *A, int nrows, int ncols) {
int ip = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
for (int i = ip; i < nrows*ncols; i += blockDim.x * gridDim.x * gridDim.y) {
A[i] = i % nrows;
}
}
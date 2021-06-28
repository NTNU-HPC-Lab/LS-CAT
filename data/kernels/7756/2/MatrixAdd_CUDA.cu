#include "includes.h"
__global__ void MatrixAdd_CUDA(int *A, int *B, int *C) {
int i= blockIdx.y*blockDim.y+ threadIdx.y;
int j = blockIdx.x*blockDim.x+ threadIdx.x;
*(C + i*N + j) =  *(A + i*N + j)+ *(B + i*N + j);

}
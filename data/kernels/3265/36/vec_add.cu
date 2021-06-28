#include "includes.h"
__global__ void vec_add(int N, int *A, int *B, int *C){
int i = threadIdx.x + blockIdx.x * blockDim.x;
if(i < N) C[0] = A[i] * B[i];
}
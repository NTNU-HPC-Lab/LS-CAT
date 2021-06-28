#include "includes.h"
__global__ void vec_add(int N, int *A, int *B, int *C){
int i = threadIdx.x + blockIdx.x * blockDim.x;
// assert( i<N );
if(i < N) C[i] = A[i] + B[i];
}
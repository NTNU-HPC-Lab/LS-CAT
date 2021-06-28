#include "includes.h"
__global__ void nodiag_normalize(float *A, float *I, int n, int i){
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

if (x < n && y < n)
if (x == i && x!=y){
I[x*n + y] /= A[i*n + i];
A[x*n + y] /= A[i*n + i];
}

}
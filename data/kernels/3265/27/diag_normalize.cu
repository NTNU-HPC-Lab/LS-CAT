#include "includes.h"
__global__ void diag_normalize(double *A, double *I, int n, int i){
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

if (x < n && y < n)
if (x == y && x == i){
I[x*n + y] /= A[i*n + i];
A[x*n + y] /= A[i*n + i];
}

}
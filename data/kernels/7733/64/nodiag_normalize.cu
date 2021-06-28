#include "includes.h"
__global__ void nodiag_normalize(double *A, double *I, int nn, int i){
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
if ( x< nn && y < nn){
if (x < nn && y < nn){
if (x == i && x!=y){
I[x*nn + y] /= A[i*nn + i];
A[x*nn + y] /= A[i*nn + i];
}
}
}
}
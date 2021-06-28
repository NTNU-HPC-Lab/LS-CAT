#include "includes.h"
__global__ void gaussjordan(double *A, double *I, int nn, int i)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
if ( x< nn && y < nn){

if (x < nn && y < nn){
if (x != i){
I[x*nn + y] -= I[i*nn + y] * A[x*nn + i];
if (y != i){
A[x*nn + y] -= A[i*nn + y] * A[x*nn + i];
}
}
}
}
}
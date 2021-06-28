#include "includes.h"
__global__ void makeProjection( float *eT, float *e, float *eigenvec, int *indices, int M, int N ) {
int elementNum = blockIdx.x * blockDim.x + threadIdx.x;
if( elementNum >= M * N ) {
return;
}
int m = elementNum / N;
int n = elementNum % N;
e[n * M + m] = eigenvec[n * M + indices[m]];
eT[m * N + n] = e[n * M + m];
}
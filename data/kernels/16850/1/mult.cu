#include "includes.h"
__global__ void mult(int* A,int* B,int* C) {
int x = threadIdx.x;
int y = threadIdx.y;

if ( x >= N || y >= M )
return;

for(int i=0,j=0; i < N && j < M ; i++, j++) {
C[x*N+y] += A[x*N+j]*B[i*N+y];
}
}
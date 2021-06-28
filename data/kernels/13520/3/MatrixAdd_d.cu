#include "includes.h"
__global__ void MatrixAdd_d( float *A, float *B, float *C, int N ) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;
int index = i * N + j;
if( ( i < N ) && ( j < N ) ) {
C[ index ] = A[ index ] + B[ index ];
}
}
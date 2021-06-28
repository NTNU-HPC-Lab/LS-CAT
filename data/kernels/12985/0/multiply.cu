#include "includes.h"

__global__ void multiply( float *A2, float *B2, float *C, int N, int threads_num ){
__shared__ float *A;
__shared__ float *B;
A = A2;	B = B2;

float tmp;
int k, pos;

int a = N * N * (blockDim.x * blockIdx.x + threadIdx.x) / threads_num, b;

if ( blockDim.x * blockIdx.x + threadIdx.x == threads_num - 1)
b = N * N;
else
b = N * N * ( blockDim.x * blockIdx.x + threadIdx.x + 1) / threads_num;

for( pos = a; pos < b; pos++ ){
tmp = 0;
for( k = 0; k < N; k++ )
tmp += A[ N * (pos / N ) + k ] * B[ k * N + pos - ( pos / N) * N];
C[ pos ] = tmp;
}
}
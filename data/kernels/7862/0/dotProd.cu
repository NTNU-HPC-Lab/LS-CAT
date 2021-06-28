#include "includes.h"
/*
#define N 512

#define N 2048
#define THREADS_PER_BLOCK 512

*/
const int THREADS_PER_BLOCK = 32;
const int N = 2048;



__global__ void dotProd( int *a, int *b, int *c ) {
__shared__ int temp[N];

temp[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];

__syncthreads(); // Evita condición de carrera.
if( 0 == threadIdx.x ) {
int sum = 0;
for(int i = 0; i < N; i++ ) {
sum += temp[i]; //lento
}
*c = sum;
}
}
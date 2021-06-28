#include "includes.h"
/*
#define N 512

#define N 2048
#define THREADS_PER_BLOCK 512

*/
const int THREADS_PER_BLOCK = 32;
const int N = 2048;



__global__ void dotProd( int *a, int *b, int *c ) {
__shared__ int temp[THREADS_PER_BLOCK];
int index = threadIdx.x + blockIdx.x * blockDim.x;

temp[threadIdx.x] = a[index] * b[index];
__syncthreads(); // Hasta que no rellenen todos los thread temp no puedo continuar...

if(threadIdx.x == 0) {
int sum = 0;
for( int i= 0; i < THREADS_PER_BLOCK; i++ ) {
sum += temp[i];
}
c[blockIdx.x] = sum;
}
}
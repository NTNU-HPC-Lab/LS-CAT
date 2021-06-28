#include "includes.h"
// curand
#define N 100
#define T 4


void llenarMatriz(int*);

__global__ void multiplicacion( int *a, int *b, int *c ) {
int i = threadIdx.x + blockIdx.x*blockDim.x; // 0 - 2047
int j = threadIdx.y + blockIdx.y*blockDim.y; // 0 - 2047

c[j+i*N] = 0; // 4,194,303

for(int k=0 ; k < N ; k++ ){
c[j+i*N] += a[k+i*N] * b[j+k*N];
}
}
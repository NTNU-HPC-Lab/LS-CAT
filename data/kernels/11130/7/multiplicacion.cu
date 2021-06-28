#include "includes.h"
__global__ void multiplicacion( int *a, int *b, int *c, int n, int m, int l ) {
int i = threadIdx.x + blockIdx.x*blockDim.x;
int j = threadIdx.y + blockIdx.y*blockDim.y;

c[j+i*l] = 0;

for(int k=0 ; k < m ; k++ ){
c[j+i*l] += a[k+i*m] * b[j+k*l];
}
}
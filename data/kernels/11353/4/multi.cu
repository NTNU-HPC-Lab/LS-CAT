#include "includes.h"
__global__ void multi(int *a, int *b, int *c,int n) {
int suma = 0;
int row = blockIdx.y * blockDim.y + threadIdx.y ;
int col = blockIdx.x * blockDim.x + threadIdx.x ;

if (row <n && col<n){
for(int i=0;i<N;++i){
suma+= a[row*n+i] * b[i*n+col];
}
}
c[row*n+col] = suma;
}
#include "includes.h"
__global__ void kernelMultMat(int *a, int *b, int *c,int m){
int i,add;

int col=blockDim.x*blockIdx.x + threadIdx.x;
int row=blockDim.y*blockIdx.y + threadIdx.y;

if(col<m && row<m) {
add=0;
for(i=0; i< m ;i++){
add += a[i+m*row]*b[col+m*i];
}
c[row*m+col] = add;
}
}
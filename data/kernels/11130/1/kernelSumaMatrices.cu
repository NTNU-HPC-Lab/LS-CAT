#include "includes.h"
__global__ void kernelSumaMatrices(float *a, float *b,int m, int n) {
int i = threadIdx.x + blockIdx.x*blockDim.x;
int j = threadIdx.y + blockIdx.y*blockDim.y;

while(i<m){
j = threadIdx.y + blockIdx.y*blockDim.y;
while(j<n){
a[i*n+j]+=b[i*n+j];
j+= blockDim.y*gridDim.y;
}
i+=blockDim.x*gridDim.x;
}
}
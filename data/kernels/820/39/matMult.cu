#include "includes.h"
__global__ void matMult(int* a, int* b, int* res,unsigned  int rows, unsigned int k, unsigned int cols){
unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;
unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;

unsigned int sum = 0;

if(r< rows && c< cols){
for(int x=0; x<k; x++){
sum += a[r*k +x] + b[x*cols + c];
}
res[r*cols + c] = sum;
}
}
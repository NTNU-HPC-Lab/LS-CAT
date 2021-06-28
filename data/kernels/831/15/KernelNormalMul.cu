#include "includes.h"
__global__ void KernelNormalMul(float *Mat1,float *Mat2,float *Mat3,int m,int n,int p){
int j = threadIdx.y + blockDim.y * blockIdx.y; // row
int i = threadIdx.x + blockDim.x * blockIdx.x; // col

if((j<m) && (i<p)){
float value=0.0;
for(int k=0;k<n;++k){
value+=Mat1[n*j+k]*Mat2[p*k+i];
}
Mat3[p*j+i]=value;
}
}
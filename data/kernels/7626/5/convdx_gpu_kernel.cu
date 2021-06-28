#include "includes.h"
__global__ void convdx_gpu_kernel(float *dx, float *dy, float *weights, const int S,const int outSize, const int inSize){
int row = blockIdx.y*blockDim.y+threadIdx.y;
int col = blockIdx.x*blockDim.x+threadIdx.x;
if(row < inSize && col < outSize){
//		printf("row %d, col %d, bias[col] %.2f\n", row, col,bias[col]);
for(int i = 0; i < S; ++i){
dx[row*outSize+col] +=dy[row* S + i ]*weights[col*S+i];
//		  printf("dy[%d] is %.1f,weight[%d] is %.1f\n", row*S+i,dy[row*S+i],col*S+i,weights[col*S+i]);
}
//		printf("conv dx %d is %3f\n",row*outSize+col, dx[row*outSize+col] );
}
}
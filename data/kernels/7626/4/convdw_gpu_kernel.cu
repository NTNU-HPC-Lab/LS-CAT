#include "includes.h"
__global__ void convdw_gpu_kernel(float *dw, float *dy, float *x, const int S,const int outSize, const int inSize){
int row = blockIdx.y*blockDim.y+threadIdx.y;
int col = blockIdx.x*blockDim.x+threadIdx.x;
if(row < inSize && col < outSize){
//		printf("row %d, col %d, bias[col] %.2f\n", row, col,bias[col]);
for(int i = 0; i < S; ++i){
dw[row*outSize+col] +=x[row+S*i ]*dy[i*outSize+col];
//		  printf("x[%d] is %.1f,dy[%d] is %.1f\n", row + S*i,x[row + S*i],i*S+row,dy[i*outSize+col]);
}
//  		printf("conv dw %d is %3f\n",row*outSize+col, dw[row*outSize+col] );
}
}
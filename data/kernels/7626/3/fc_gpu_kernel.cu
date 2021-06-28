#include "includes.h"
__global__ void fc_gpu_kernel(float *y, float *x, float *weights, const int weightHeight,const int outSize, const int inSize){
//printf(x);
int row = blockIdx.y*blockDim.y+threadIdx.y;
int col = blockIdx.x*blockDim.x+threadIdx.x;
//printf("row %d, col %d in fc.cu \n",row,col);
if(row < inSize && col < outSize){
//float acc = 0;
for(int i = 0; i < weightHeight; ++i){
y[row*outSize+col] +=x[row*weightHeight + i ]*weights[i*outSize+col];
//printf("x[%d] is %.1f,weight[%d] is %.1f\n", row*weightHeight+i,x[row*weightHeight+i],i*outSize+col,weights[i*outSize+col]);
}
//printf("acc is %3f, y %d is %3f\n",acc, row*outSize+col, y[row*outSize+col] );
}
}
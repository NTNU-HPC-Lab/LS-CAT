#include "includes.h"
__global__ void padding(int *op,int *ip,int N,int C,int H,int W,int Py,int Px){
unsigned int input_id = (blockIdx.x*gridDim.y + blockIdx.y + blockIdx.z*gridDim.x*gridDim.y)*blockDim.x + threadIdx.x;
int i = input_id/(C*H*W);
input_id = input_id%(C*H*W);
int j = input_id/(H*W);
input_id = input_id%(H*W);
int k = input_id/W;
int l = input_id%W;
*(op + i*C*(H + 2*Py)*(W + 2*Px) + j*(H + 2*Py)*(W + 2*Px) + (k + Py)*(W + 2*Px) + (l + Px)) = *(ip + i*C*H*W + j*H*W + k*W + l);
}
#include "includes.h"
__global__ void RELU(int *ip, int N, int C, int H, int W){
unsigned int input_id = blockDim.x*blockIdx.x + threadIdx.x;
int i = input_id/(C*H*W);
input_id = input_id%(C*H*W);
int j = input_id/(H*W);
input_id = input_id%(H*W);
int k = input_id/(W);
int l = input_id%W;

int temp = *(ip + i*C*H*W + j*H*W + k*W + l);
if(temp<0)
*(ip + i*C*H*W + j*H*W + k*W + l) = 0;

}
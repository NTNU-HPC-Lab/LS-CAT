#include "includes.h"
__global__ void softmax_linear(float* softmaxP, float* b, int rows, int cols){
int tid = threadIdx.x;
int bid = blockIdx.x;

float _max = -100000000.0;
float sum = 0.0;

extern __shared__ float _share[];

if(tid * cols + bid < rows * cols){
for(int i = 0 ; i < rows ; i++) _share[i] = b[i * cols + bid];
for(int i = 0 ; i < rows ; i++)	_max = max(_max, _share[i]);
for(int i = 0 ; i < rows ; i++)	_share[i] = __expf(_share[i]-_max);
for(int i = 0 ; i < rows ; i++)	sum += _share[i];
for(int i = 0 ; i < rows ; i++)	softmaxP[i * cols + bid] = _share[i]/sum;
}
}
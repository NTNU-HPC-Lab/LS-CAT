#include "includes.h"
__global__ void prefixSumBackward(float* arr,int step){

int bx = blockIdx.x;
int tx = threadIdx.x;

int BX = blockDim.x;

int i = bx*BX+tx;

int ii = i+1;

if(i >= n || ii > n/float(step)) return;

int temp = arr[ii*step-1];
arr[ii*step-1]	 += arr[ii*step-step/2-1];
arr[ii*step-step/2-1] = temp;

}
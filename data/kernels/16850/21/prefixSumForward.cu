#include "includes.h"
__global__ void prefixSumForward(float* arr,int step){

int bx = blockIdx.x;
int tx = threadIdx.x;

int BX = blockDim.x;

int i = bx*BX+tx;

int ii = i+1;

if( ii <= n &&  ii > n/float(step)) return;

arr[ii*step-1] += arr[ii*step-step/2-1];

if(step==n && n-1 == ii*step-1) {
arr[ii*step]  = arr[ii*step-1];
arr[ii*step-1]= 0;
}
}
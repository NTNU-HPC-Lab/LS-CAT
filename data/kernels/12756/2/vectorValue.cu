#include "includes.h"
__global__ void vectorValue (float *a, float *b, int n){
int i= blockDim.x * blockIdx.x + threadIdx.x;

if (i<n){
a[i]=threadIdx.x*2;
b[i]=threadIdx.x;
}

}
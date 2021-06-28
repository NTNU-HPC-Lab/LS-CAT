#include "includes.h"
__global__ void vectorAddition (float *a, float *b, float *c, int n){
int i= blockDim.x * blockIdx.x + threadIdx.x;

if (i<n){
c[i] = a[i]+b[i];
}

}
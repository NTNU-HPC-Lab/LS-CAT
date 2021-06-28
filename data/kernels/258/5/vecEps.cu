#include "includes.h"
__global__ void vecEps(float* a,const int N){
int i = blockIdx.x*blockDim.x + threadIdx.x;
if(a[i] < EPS && i < N)
a[i] = EPS;
}
#include "includes.h"

#define N 10000000
#define MAX_ERR 1e-6


__global__ void vector_add(float* out,float* a,float* b,int n){
int index = threadIdx.x;
int stride = blockDim.x;
for(int i=index ; i<n ;i=i+stride){
out[i]=a[i]+b[i];
}
}
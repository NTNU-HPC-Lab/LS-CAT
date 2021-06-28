#include "includes.h"

extern "C" {
}


__global__ void weighted_delta_kernel(int n, float *a, float *b, float *s, float *da, float *db, float *ds, float *dc)
{
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if(i < n){
if(da) da[i] += dc[i] * s[i];
db[i] += dc[i] * (1-s[i]);
ds[i] += dc[i] * a[i] + dc[i] * -b[i];
}
}
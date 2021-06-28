#include "includes.h"
__global__ void l2norm_kernel(int N, float *x, float *dx, int batch, int filters, int spatial)
{
int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (index >= N) return;
int b = index / spatial;
int i = index % spatial;
int f;
float sum = 0;
for(f = 0; f < filters; ++f){
int index = b*filters*spatial + f*spatial + i;
sum += powf(x[index], 2);
}
sum = sqrtf(sum);
if(sum == 0) sum = 1;
//printf("%f\n", sum);
for(f = 0; f < filters; ++f){
int index = b*filters*spatial + f*spatial + i;
x[index] /= sum;
dx[index] = (1 - x[index]) / sum;
}
}
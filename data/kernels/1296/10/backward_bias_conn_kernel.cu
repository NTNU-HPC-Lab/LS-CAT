#include "includes.h"
__global__ void backward_bias_conn_kernel(float *bias_updates, float *delta, int batch, int n)
{
int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (index >= n) return;
int b;
float sum = 0;
for(b = 0; b < batch; ++b){
int i = b*n + index;
sum += delta[i];
}
bias_updates[index] += sum;
}
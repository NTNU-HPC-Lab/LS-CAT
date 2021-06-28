#include "includes.h"
__global__ void forward_softmax_layer_kernel(int n, int batch, float *input, float temp, float *output)
{
int b = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if(b >= batch) return;

int i;
float sum = 0;
float largest = -INFINITY;
for(i = 0; i < n; ++i){
int val = input[i+b*n];
largest = (val>largest) ? val : largest;
}
for(i = 0; i < n; ++i){
sum += exp(input[i+b*n]/temp-largest/temp);
}
sum = (sum != 0) ? largest/temp+log(sum) : largest-100;
for(i = 0; i < n; ++i){
output[i+b*n] = exp(input[i+b*n]/temp-sum);
}
}
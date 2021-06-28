#include "includes.h"
__global__ void _bcnn_forward_softmax_layer_kernel(int n, int batch, float *input, float *output)
{
int i;
float sum = 0;
float largest = -INFINITY;
int b = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

if (b >= batch) {
return;
}

for (i = 0; i < n; ++i) {
int val = input[i+b*n];
largest = (val>largest) ? val : largest;
}

for (i = 0; i < n; ++i) {
sum += exp(input[i+b*n]-largest);
}

sum = (sum != 0) ? largest+log(sum) : largest-100;

for (i = 0; i < n; ++i) {
output[i+b*n] = exp(input[i+b*n]-sum);
}
}
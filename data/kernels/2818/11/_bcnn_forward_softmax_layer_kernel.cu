#include "includes.h"
__global__ void _bcnn_forward_softmax_layer_kernel(int n, int batch, float *input, float *output) {
float sum = 0.f;
float maxf = -INFINITY;
int b = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

if (b >= batch) {
return;
}
for (int i = 0; i < n; ++i) {
int val = input[i + b * n];
maxf = (val > maxf) ? val : maxf;
}
for (int i = 0; i < n; ++i) {
sum += exp(input[i + b * n] - maxf);
}
sum = (sum != 0) ? maxf + log(sum) : maxf - 100.f;
for (int i = 0; i < n; ++i) {
output[i + b * n] = exp(input[i + b * n] - sum);
}
}
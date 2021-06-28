#include "includes.h"
__global__ void ApplySecondPool(float* input, float* output){
int id = threadIdx.x + blockIdx.x * blockDim.x;

for (int i = 0; i < 73; ++i){
//float total = input[i * 2 + id * 2 * 296] +
//	input[i * 2 + 1 + id * 2 * 296] + input[i * 2 + id * 2 * 296 + 296] + input[i * 2 + 1 + id * 2 * 296 + 296];
//total /= 4;

float total = 0;
total = max(	   input[i * 2 + id * 2 * 146],
input[i * 2 + id * 2 * 146 + 1]);
total = max(total, input[i * 2 + id * 2 * 146 + 146]);
total = max(total, input[i * 2 + id * 2 * 146 + 146 + 1]);

output[i + id * 73] = total;//((float)i) / 73.0f;
output[i + id * 73] = 1 / (1 + exp(-(output[i + id * 73] * 2 - 1)));
}
}
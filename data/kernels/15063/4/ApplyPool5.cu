#include "includes.h"
__global__ void ApplyPool5(float* input, float* output){
int id = threadIdx.x + blockIdx.x * blockDim.x;

for (int i = 0; i < 148; ++i){
//float total = input[i * 2 + id * 2 * 296] +
//	input[i * 2 + 1 + id * 2 * 296] + input[i * 2 + id * 2 * 296 + 296] + input[i * 2 + 1 + id * 2 * 296 + 296];
//total /= 4;

float total = 0;
total = max(	   input[i * 2 + id * 2 * 296],
input[i * 2 + id * 2 * 296 + 1]);
total = max(total, input[i * 2 + id * 2 * 296 + 296]);
total = max(total, input[i * 2 + id * 2 * 296 + 296 + 1]);

//float total = ((float)i) / 148.0f; // input[i * 2 + id * 2 * 296];

//if (total < -0.1f){
//	printf("ApplyFirstPool total: %f\n", total);
//}
output[i + id * 148] = total;
output[i + id * 148] = 1 / (1 + exp(-(output[i + id * 148] * 2 - 1)));
}
}
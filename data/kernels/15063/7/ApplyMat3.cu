#include "includes.h"
__global__ void ApplyMat3(float* input, float* output, float* matrix){
int id = threadIdx.x + blockDim.x * blockIdx.x;

//for (int i = 0; i < 148 * 148; ++i){
//	if(input[i] > 0.1f) printf("Input above 0, %i", i);
//}

for (int i = 0; i < 146; ++i){
float total = 0.0f;

//if (input[id * 148 + i] > 0.1f) printf("Input above 0, %i", id * 148 + i);

total += input[id * 148 + i] * matrix[0];
total += input[id * 148 + i + 1] * matrix[1];
total += input[id * 148 + i + 2] * matrix[2];

total += input[id * 148 + i + 148 * 1] * matrix[3];
total += input[id * 148 + i + 148 * 1 + 1] * matrix[4];
total += input[id * 148 + i + 148 * 1 + 2] * matrix[5];

total += input[id * 148 + i + 148 * 2] * matrix[6];
total += input[id * 148 + i + 148 * 2 + 1] * matrix[7];
total += input[id * 148 + i + 148 * 2 + 2] * matrix[8];

//if (total < -0.1f || total > 0.1f) printf("Total: %f", total);

total = fmax(0.0f, total);

output[i + id * 146] = total;
}
}
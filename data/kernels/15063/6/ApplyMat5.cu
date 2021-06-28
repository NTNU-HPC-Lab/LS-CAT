#include "includes.h"
__global__ void ApplyMat5(float* input, float* output, float* matrix){
int id = threadIdx.x + blockDim.x * blockIdx.x;

for (int i = 0; i < 296; ++i){
float total = 0.0f;
total += input[id * 300 + i] * matrix[0];
total += input[id * 300 + i + 1] * matrix[1];
total += input[id * 300 + i + 2] * matrix[2];
total += input[id * 300 + i + 3] * matrix[3];
total += input[id * 300 + i + 4] * matrix[4];

total += input[id * 300 + i + 300 * 1] * matrix[5];
total += input[id * 300 + i + 300 * 1 + 1] * matrix[6];
total += input[id * 300 + i + 300 * 1 + 2] * matrix[7];
total += input[id * 300 + i + 300 * 1 + 3] * matrix[8];
total += input[id * 300 + i + 300 * 1 + 4] * matrix[9];

total += input[id * 300 + i + 300 * 2] * matrix[10];
total += input[id * 300 + i + 300 * 2 + 1] * matrix[11];
total += input[id * 300 + i + 300 * 2 + 2] * matrix[12];
total += input[id * 300 + i + 300 * 2 + 3] * matrix[13];
total += input[id * 300 + i + 300 * 2 + 4] * matrix[14];

total += input[id * 300 + i + 300 * 3] * matrix[15];
total += input[id * 300 + i + 300 * 3 + 1] * matrix[16];
total += input[id * 300 + i + 300 * 3 + 2] * matrix[17];
total += input[id * 300 + i + 300 * 3 + 3] * matrix[18];
total += input[id * 300 + i + 300 * 3 + 4] * matrix[19];

total += input[id * 300 + i + 300 * 4] * matrix[20];
total += input[id * 300 + i + 300 * 4 + 1] * matrix[21];
total += input[id * 300 + i + 300 * 4 + 2] * matrix[22];
total += input[id * 300 + i + 300 * 4 + 3] * matrix[23];
total += input[id * 300 + i + 300 * 4 + 4] * matrix[24];

total = fmax(0.0f, total);

output[i + id * 296] = total;
}
}
#include "includes.h"
__global__ void kSwapColumns(float* source, float* target, float* indices1, float* indices2, int cols, int width, int height){
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
float temp;
unsigned int column, row, source_pos, target_pos;
for (unsigned int i = idx; i < height * cols; i += numThreads) {
column = i / height;
row = i % height;
source_pos = height * (int)indices1[column] + row;
target_pos = height * (int)indices2[column] + row;
temp = source[source_pos];
source[source_pos] = target[target_pos];
target[target_pos] = temp;
}
}
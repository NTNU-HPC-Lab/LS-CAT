#include "includes.h"
__global__ void kShuffleColumns(float* source, float* target, float* indices, int width, int height){
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
float temp1, temp2;
unsigned int column, row, pos1, pos2;

for (unsigned int i = idx; i < height * ((width+1) / 2); i += numThreads) {
column = 2 * (i / height);
row = i % height;
if (column + 1 >= width) {
pos1 = height * (int)indices[column] + row;
target[pos1] = source[pos1];
} else {
pos1 = height * (int)indices[column] + row;
pos2 = height * (int)indices[column + 1] + row;
temp1 = source[pos1];
temp2 = source[pos2];
target[pos2] = temp1;
target[pos1] = temp2;
}
}
}
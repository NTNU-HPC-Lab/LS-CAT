#include "includes.h"
__global__ void kAccumulateColumns(float* mat, float* indices, float* target, int mat_width, int target_width, int height, float mult, int avg){
const int row = gridDim.x * blockIdx.y + blockIdx.x;
const int column = threadIdx.x;
if (row < height && column < target_width) {
float cur_sum = 0.0;
unsigned int count = 0;
for (unsigned int i = 0; i < mat_width; i ++) {
count += ((int)indices[i] == column) ? 1 : 0 ;
cur_sum += ((int)indices[i] == column) ? mat[row + i * height] : 0 ;
}
target[row + height * column] = mult * cur_sum / ((avg == 1 && count > 0) ? count : 1);
}
}
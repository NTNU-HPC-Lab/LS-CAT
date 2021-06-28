#include "includes.h"
__global__ void addValue(int * array_val, int*b_array_val) {
int x = threadIdx.x;
int sum = 0;

for(unsigned int i = 0; i < ROWS; i++) {
sum += array_val[i*COLUMNS+x];
}
b_array_val[x] = sum;
}
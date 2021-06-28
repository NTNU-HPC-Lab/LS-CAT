#include "includes.h"
__global__ void counting_sort(int* array, int *temp, int size) {
int i, j, count;
i = threadIdx.x + (blockIdx.x * blockDim.x);
if (i < size) {
count = 0;
for(j = 0; j < size; j++) {
if(array[j] < array[i]) {
count++;
} else if(array[i] == array[j] && j < i) {
count++;
}
}
temp[count] = array[i];
}
}
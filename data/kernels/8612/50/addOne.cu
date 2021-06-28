#include "includes.h"
__global__ void addOne(int* array, int size) {
if ( blockIdx.x < size ) {
array[blockIdx.x]++;
}
}
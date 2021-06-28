#include "includes.h"

using namespace std;
#ifndef MAP_FILE
#define MAP_FILE MAP_SHARED
#endif

__global__ void arrayFill_greg(float* data, float value, int size) {
int stride = gridDim.x * blockDim.x;
int tid = threadIdx.x + blockIdx.x * blockDim.x;
for (int i = tid; i < size; i += stride) data[i] = value;
}
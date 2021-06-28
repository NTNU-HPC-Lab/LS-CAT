#include "includes.h"

using namespace std;
#ifndef MAP_FILE
#define MAP_FILE MAP_SHARED
#endif

__global__ void hardTanh(float* in, float* out, float min_val, float max_val, int size) {
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int stride = gridDim.x * blockDim.x;
for (int i = tid; i < size; i += stride) {
out[i] = in[i] < min_val ? min_val : (in[i] > max_val ? max_val : in[i]);
}
}
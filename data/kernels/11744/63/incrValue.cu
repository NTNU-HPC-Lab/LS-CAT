#include "includes.h"
__global__ void incrValue(float *data, int idx, float value) {
if(threadIdx.x == 0  && blockIdx.x == 0) {
data[idx] += value;
}
}
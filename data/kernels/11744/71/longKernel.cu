#include "includes.h"
__global__ void longKernel(float *data, int N, float value) {
for(int i = 0; i < N; i++) {
data[i] += value;
}
}
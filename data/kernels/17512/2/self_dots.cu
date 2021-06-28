#include "includes.h"
__global__ void self_dots(int n, int d, double* data, double* dots) {
double accumulator = 0;
int global_id = blockDim.x * blockIdx.x + threadIdx.x;

if (global_id < n) {
for (int i = 0; i < d; i++) {
double value = data[i + global_id * d];
accumulator += value * value;
}
dots[global_id] = accumulator;
}
}
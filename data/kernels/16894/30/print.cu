#include "includes.h"
__global__ void print() { printf("GPU thread %d\n", threadIdx.x); }
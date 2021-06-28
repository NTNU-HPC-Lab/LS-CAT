#include "includes.h"


__global__ void print_gpu(void) {
printf("Houston, we have a problem in section [%d,%d] \
From Apollo 13\n", threadIdx.x,blockIdx.x);
}
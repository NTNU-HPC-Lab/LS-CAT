#include "includes.h"
__global__ void __veccmp(int *a, int *b, int *d) {
printf("__veccmp() not defined for CUDA Arch < 300\n");
}
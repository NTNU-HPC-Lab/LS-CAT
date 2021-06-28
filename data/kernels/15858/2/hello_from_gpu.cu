#include "includes.h"
__global__ void hello_from_gpu()
{
const int b = blockIdx.x;
const int tx = threadIdx.x;
const int ty = threadIdx.y;
printf("Hello World from block-%d and thread-(%d, %d)!\n", b, tx, ty);
}
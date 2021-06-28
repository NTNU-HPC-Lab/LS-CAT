#include "includes.h"
__global__ void DrawObstacles(uchar4 *ptr, int* indices, int size) {

int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

while (thread_id < size) {
int index = indices[thread_id];
ptr[index].x = 0;
ptr[index].y = 0;
ptr[index].z = 0;
ptr[index].w = 255;

thread_id += blockDim.x * gridDim.x;
}
}
#include "includes.h"
__global__ void assignColIds(int* colIds, const int* colOffsets) {
int myId = blockIdx.x;
int start = colOffsets[myId];
int end = colOffsets[myId + 1];
for (int id = start + threadIdx.x; id < end; id += blockDim.x) {
colIds[id] = myId;
}
}
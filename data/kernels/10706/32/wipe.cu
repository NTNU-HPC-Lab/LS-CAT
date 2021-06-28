#include "includes.h"
__global__ void wipe(int *buffer, int length) {
length >>= 5;
int tid = threadIdx.x;
for(int i = 0; i < length; i++) {
buffer[(i << 5) + tid] = -1;
}
}
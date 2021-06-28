#include "includes.h"
__global__ void kernCopyToFrame(int N, uint8_t * frame, float * src) {
int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
if (idx >= N) {
return;
}
if (src[idx] < 0) {
frame[idx] = 0;
} else {
frame[idx] = (uint8_t) src[idx];
}
return;
}
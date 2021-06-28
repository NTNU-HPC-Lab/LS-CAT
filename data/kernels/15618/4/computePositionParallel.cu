#include "includes.h"
__global__ void computePositionParallel(float *agentsX, float *agentsY, float *destX, float *destY, float *destR, int n, int *reached) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;

for (int i = index; i < n; i += stride) {
// if there is no destination to go to
if (destX[i] == -1 || destY[i] == -1) {
continue;
}

// compute and update next position
double diffX = destX[i] - agentsX[i];
double diffY = destY[i] - agentsY[i];
double length = sqrtf(diffX * diffX + diffY * diffY);
agentsX[i] = (float)llrintf(agentsX[i] + diffX / length);
agentsY[i] = (float)llrintf(agentsY[i] + diffY / length);

// check if next position is inside the destination radius
diffX = destX[i] - agentsX[i];
diffY = destY[i] - agentsY[i];
length = sqrtf(diffX * diffX + diffY * diffY);

if (length < destR[i]) {
reached[i] = 1;
}
}
}
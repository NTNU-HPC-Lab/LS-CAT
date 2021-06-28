#include "includes.h"
__global__ void update(float* original, float* newTE, float* current, int nhalf) {
int i = threadIdx.x + blockDim.x*blockIdx.x;
if (i < nhalf) {
current[i] /= nhalf;
newTE[i] = (original[i] < current[i]) ? current[i] : original[i];   // LIKELY, THERE IS A PERFORMANCE LOSS HERE
}
}
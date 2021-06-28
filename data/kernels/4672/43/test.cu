#include "includes.h"
__global__ void test(float* nonSmoothed, float* smoothed, int* mask, int nhalf) {
int i = threadIdx.x + blockDim.x*blockIdx.x;
int diff;
if (i < nhalf) {
diff = fabs(nonSmoothed[i] - smoothed[i]/nhalf);
mask[i] = (diff > 0.23) ? 1 : 0;   // WHAT THRESHOLD TO USE?? different behaviour as opposed to CPU version!
}
}
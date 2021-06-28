#include "includes.h"
__global__ void calcCDFnormalized(const unsigned int *histo, float *cdf, size_t width, size_t height) {
for (int i = 0; i <= threadIdx.x; i++) {
cdf[threadIdx.x] += (float) histo[i];
}
cdf[threadIdx.x] *= 1.0f / float((width * height));
}
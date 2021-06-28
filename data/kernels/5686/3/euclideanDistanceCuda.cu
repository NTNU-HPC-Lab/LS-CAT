#include "includes.h"
__global__ void euclideanDistanceCuda(float3* pDotProducts, size_t pSize, float* results) {
int instance = blockIdx.x * blockDim.x + threadIdx.x;

while (instance < pSize) {
results[instance] = pDotProducts[instance].x - 2*pDotProducts[instance].y + pDotProducts[instance].z;
if (results[instance] < 0.0) results[instance] = 0.0;
instance += gridDim.x;
}
}
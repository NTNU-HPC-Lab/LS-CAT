#include "includes.h"
__global__ void cosineSimilarityCuda(float3* pDotProducts, size_t pSize, float* results) {
int instance = blockIdx.x * blockDim.x + threadIdx.x;

while (instance < pSize) {
results[instance] = pDotProducts[instance].y / (sqrtf(pDotProducts[instance].x)* sqrtf(pDotProducts[instance].z));
instance += gridDim.x;
}
}
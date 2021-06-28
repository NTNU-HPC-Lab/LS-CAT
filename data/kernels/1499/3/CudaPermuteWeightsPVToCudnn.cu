#include "includes.h"
__global__ void CudaPermuteWeightsPVToCudnn( float *dest, float *src, int outFeatures, int ny, int nx, int inFeatures, int manyScaleX, int manyScaleY) {
// Parameter dimensions are PV source dimensions
int kSrc = (blockIdx.x * blockDim.x) + threadIdx.x;
if (kSrc < outFeatures * manyScaleX * manyScaleY * ny * nx * inFeatures) {
int kOF = kSrc / (ny * nx * inFeatures);
int kY  = (kSrc % (ny * nx * inFeatures)) / (nx * inFeatures);
int kX  = (kSrc % (nx * inFeatures)) / inFeatures;
int kIF = (kSrc % inFeatures);

int sOF = inFeatures * ny * nx;
int sIF = ny * nx;
int sY  = nx;

int kDest = kOF * sOF + kIF * sIF + (ny - kY - 1) * sY + (nx - kX - 1);

dest[kDest] = src[kSrc];
}
}
#include "includes.h"
__global__ void CudaPermuteCudnnToPV( float *dest, float *src, int outFeatures, int ny, int nx, int inFeatures, int manyScaleX, int manyScaleY) {
// parameter dimensions are in dest PV format
int srcNx         = nx / manyScaleX;
int srcNy         = ny / manyScaleY;
int srcInFeatures = inFeatures * manyScaleX * manyScaleY;

int kDest = (blockIdx.x * blockDim.x) + threadIdx.x;
if (kDest < outFeatures * ny * nx * inFeatures) {
int kOF = kDest / (ny * nx * inFeatures);
int kY  = (kDest % (ny * nx * inFeatures)) / (nx * inFeatures);
int kX  = (kDest % (nx * inFeatures)) / inFeatures;
int kIF = (kDest % inFeatures);

// Recalculate x, y, and f based on manyScale
kIF = kIF + inFeatures * (kX % manyScaleX + (kY % manyScaleY) * manyScaleX);
kX  = kX / manyScaleX;
kY  = kY / manyScaleY;

int sOF = srcInFeatures * srcNy * srcNx;
int sIF = srcNy * srcNx;
int sY  = srcNx;

int kSrc = kOF * sOF + kIF * sIF + kY * sY + kX;

dest[kDest] = src[kSrc];
}
}
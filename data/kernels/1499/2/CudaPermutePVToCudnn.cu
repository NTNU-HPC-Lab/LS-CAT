#include "includes.h"
__global__ void CudaPermutePVToCudnn( float *dest, float *src, int outFeatures, int ny, int nx, int inFeatures, int manyScaleX, int manyScaleY, int cropX, int cropY) {
// parameter dimensions are in source PV format
int destNx         = (nx - 2 * cropX) / manyScaleX;
int destNy         = (ny - 2 * cropY) / manyScaleY;
int destInFeatures = inFeatures * manyScaleX * manyScaleY;

int kSrc = (blockIdx.x * blockDim.x) + threadIdx.x;
if (kSrc < outFeatures * ny * nx * inFeatures) {
int kOF = kSrc / (ny * nx * inFeatures);
int kY  = (kSrc % (ny * nx * inFeatures)) / (nx * inFeatures);
int kX  = (kSrc % (nx * inFeatures)) / inFeatures;
int kIF = (kSrc % inFeatures);

// check if in bounds
if (kX < cropX || kX >= nx - cropX) {
return;
}
else {
kX = kX - cropX;
}
if (kY < cropY || kY >= ny - cropY) {
return;
}
else {
kY = kY - cropY;
}

// Recalculate x, y, and f based on manyScale
kIF = kIF + inFeatures * (kX % manyScaleX + (kY % manyScaleY) * manyScaleX);
kX  = kX / manyScaleX;
kY  = kY / manyScaleY;

int sOF = destInFeatures * destNy * destNx;
int sIF = destNy * destNx;
int sY  = destNx;

int kDest = kOF * sOF + kIF * sIF + kY * sY + kX;

dest[kDest] = src[kSrc];
}
}
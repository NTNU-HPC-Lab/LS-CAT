#include "includes.h"
__global__ void CudaPermuteWeightsPVToCudnn( float *dest, float *src, int numArbors, int outFeatures, int ny, int nx, int inFeatures) {
// Parameter dimensions are PV source dimensions
int kSrc = (blockIdx.x * blockDim.x) + threadIdx.x;
if (kSrc < outFeatures * ny * nx * inFeatures) {
int kA  = kSrc / (outFeatures * ny * nx * inFeatures);
int kOF = (kSrc % (outFeatures * ny * nx * inFeatures)) / (ny * nx * inFeatures);
int kY  = (kSrc % (ny * nx * inFeatures)) / (nx * inFeatures);
int kX  = (kSrc % (nx * inFeatures)) / inFeatures;
int kIF = (kSrc % inFeatures);

int sA  = outFeatures * inFeatures * ny * nx;
int sOF = inFeatures * ny * nx;
int sIF = ny * nx;
int sY  = nx;

int kDest = kA * sA + kOF * sOF + kIF * sIF + (ny - kY - 1) * sY + (nx - kX - 1);

dest[kDest] = src[kSrc];
}
}
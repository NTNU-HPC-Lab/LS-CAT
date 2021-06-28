#include "includes.h"
__device__ float computeS(float *sumTable, int rowNumberN, int colNumberM, int startX, int startY, int Kx, int Ky) {
startX--;
startY--;
float S =
sumTable[startX + Kx + (Ky + startY) * colNumberM] -
(startX < 0 ? 0 : sumTable[startX + (Ky + startY) * colNumberM]) -
(startY < 0 ? 0 : sumTable[startX + Kx + startY * colNumberM]) +
(startX < 0 || startY < 0 ? 0 : sumTable[startX + startY * colNumberM]);
return S;
}
__global__ void calculateFeatureDifference(float *templateFeatures, int colNumberM, int rowNumberN, float *l1SumTable, float *l2SumTable, float *lxSumTable, float *lySumTable, int Kx, int Ky, float *differences) {
int widthLimit = colNumberM - Kx + 1;
int heightLimit = rowNumberN - Ky + 1;

float meanVector;
float varianceVector;
float xGradientVector;
float yGradientVector;
int startX = threadIdx.x + blockIdx.x * blockDim.x;
int startY = threadIdx.y + blockIdx.y * blockDim.y;
if (startX >= widthLimit || startY >= heightLimit) return;
float S1D =
computeS(l1SumTable, rowNumberN, colNumberM, startX, startY, Kx, Ky);
float S2D =
computeS(l2SumTable, rowNumberN, colNumberM, startX, startY, Kx, Ky);

meanVector = S1D / (Kx * Ky);

varianceVector = S2D / (Kx * Ky) - powf(meanVector, 2);

float SxD =
computeS(lxSumTable, rowNumberN, colNumberM, startX, startY, Kx, Ky);

xGradientVector = 4 * (SxD - (startX + Kx / 2.0) * S1D) / (Kx * Kx * Ky);

float SyD =
computeS(lySumTable, rowNumberN, colNumberM, startX, startY, Kx, Ky);
yGradientVector = 4 * (SyD - (startY + Ky / 2.0) * S1D) / (Ky * Ky * Kx);

differences[startX + startY * widthLimit] = norm4df(
templateFeatures[0] - meanVector, templateFeatures[1] - varianceVector,
templateFeatures[2] - xGradientVector,
templateFeatures[3] - yGradientVector);
}
#include "includes.h"
__global__ void cuda_Cal_Gfw(float *GfW, float2 *Grf, float2 *Gcf, int nRows, int nCols) {
unsigned int Tidx = threadIdx.x + blockIdx.x * blockDim.x;
unsigned int Tidy = threadIdx.y + blockIdx.y * blockDim.y;
unsigned int index;

float GfW_temp;
float2 Grf_temp, Gcf_temp;

if ((Tidx < nCols) && (Tidy < nRows)) {

index = Tidx + Tidy * nCols;

Grf_temp = Grf[index];
Gcf_temp = Gcf[index];
GfW_temp = Grf_temp.x * Grf_temp.x + Grf_temp.y * Grf_temp.y +
Gcf_temp.x * Gcf_temp.x + Gcf_temp.y * Gcf_temp.y;

GfW[index] = GfW_temp;
}
}
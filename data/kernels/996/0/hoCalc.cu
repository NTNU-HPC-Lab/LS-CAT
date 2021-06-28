#include "includes.h"


__global__ void hoCalc(double* rn, double* soilHeat, double* ho, int width_band) {

int col = threadIdx.x + blockIdx.x * blockDim.x;

while (col < width_band) {

ho[col] = rn[col] - soilHeat[col];

col += blockDim.x * gridDim.x;

}

}
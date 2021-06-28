#include "includes.h"
__global__ void GetScaleFactorsKernel(float *indata, float *base, float *stdev, float *factors, int nchans, int processed) {

// NOTE: Filterbank file format coming in
//float mean = indata[threadIdx.x];
float mean = 0.0f;
// NOTE: Depending whether I save STD or VAR at the end of every run
// float estd = stdev[threadIdx.x];
float estd = stdev[threadIdx.x] * stdev[threadIdx.x] * (processed - 1.0f);
float oldmean = base[threadIdx.x];

//float estd = 0.0f;
//float oldmean = 0.0;

float val = 0.0f;
float diff = 0.0;
for (int isamp = 0; isamp < 2 * NACCUMULATE; ++isamp) {
val = indata[isamp * nchans + threadIdx.x];
diff = val - oldmean;
mean = oldmean + diff * factors[processed + isamp + 1];
estd += diff * (val - mean);
oldmean = mean;
}
base[threadIdx.x] = mean;
stdev[threadIdx.x] = sqrtf(estd / (float)(processed + 2 * NACCUMULATE - 1.0f));
// stdev[threadIdx.x] = estd;
}
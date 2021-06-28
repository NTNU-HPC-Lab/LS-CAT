#include "includes.h"
__global__ void cunn_LookupTable_accGradParametersKernel( float *input, float *indices, float *gradOutput, float *gradWeight, float *count, float defaultScale, long numel, long stride, int paddingValue) {

int idx = blockIdx.x * 4 + threadIdx.y;

// Each warp is responsible for an input into the LookupTable.
// If the preceeding input has the same as this input, then the warp
// exits immediately. The warp also processes subsequent inputs with the
// same value.
//
// Input Warp
// 1     <warp 1>
// 1     <warp 1> (<warp 2> exits without doing any work)
// 5     <warp 3>
// 8     <warp 4>

// Number of values proceessed by each thread (grain size)
const int SZ = 4;

if (idx < numel
&& (idx == 0 || input[idx] != input[idx - 1])
&& input[idx] != paddingValue) {
do {
const int startFeature = threadIdx.x + blockIdx.y * blockDim.x * SZ;
const int weightRow = ((int) input[idx] - 1) * stride;
const int gradOutputRow = ((int) indices[idx] - 1) * stride;
const float scale = count ? defaultScale / count[idx] : defaultScale;

float gradient[SZ];
float weight[SZ];

#pragma unroll
for (int ii = 0; ii < SZ; ii++)
{
int featureDim = startFeature + ii * WARP_SIZE;
if (featureDim < stride)
{
gradient[ii] = gradOutput[gradOutputRow + featureDim];
weight[ii] = gradWeight[weightRow + featureDim];
}
}

#pragma unroll
for (int ii = 0; ii < SZ; ii++)
{
weight[ii] += gradient[ii] * scale;
}

#pragma unroll
for (int ii = 0; ii < SZ; ii++)
{
int featureDim = startFeature + ii * WARP_SIZE;
if (featureDim < stride)
{
gradWeight[weightRow + featureDim] = weight[ii];
}
}

idx++;
} while (idx < numel && input[idx] == input[idx - 1]);
}
}
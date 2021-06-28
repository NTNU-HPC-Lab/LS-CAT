#include "includes.h"
__global__ void combine_kernel(int nPixels, int cuePitchInFloats, float* devBg, float* devCga, float* devCgb, float* devTg, float* devMpb, float* devCombinedg) {
int index = blockDim.x * blockIdx.x + threadIdx.x;
int orientation = threadIdx.y;
int orientedIndex = orientation * cuePitchInFloats + index;
if (index < nPixels) {
float accumulant = 0.0;
float accumulant2=0.0;
float* pointer = &devBg[orientedIndex];
accumulant += *pointer * coefficients[0];
accumulant2 += *pointer * weights[0];
pointer += 8 * cuePitchInFloats;
accumulant += *pointer * coefficients[1];
accumulant2 += *pointer * weights[1];
pointer += 8 * cuePitchInFloats;
accumulant += *pointer * coefficients[2];
accumulant2 += *pointer * weights[2];
pointer = &devCga[orientedIndex];
accumulant += *pointer * coefficients[3];
accumulant2 += *pointer * weights[3];
pointer += 8 * cuePitchInFloats;
accumulant += *pointer * coefficients[4];
accumulant2 += *pointer * weights[4];
pointer += 8 * cuePitchInFloats;
accumulant += *pointer * coefficients[5];
accumulant2 += *pointer * weights[5];
pointer = &devCgb[orientedIndex];
accumulant += *pointer * coefficients[6];
accumulant2 += *pointer * weights[6];
pointer += 8 * cuePitchInFloats;
accumulant += *pointer * coefficients[7];
accumulant2 += *pointer * weights[7];
pointer += 8 * cuePitchInFloats;
accumulant += *pointer * coefficients[8];
accumulant2 += *pointer * weights[8];
pointer = &devTg[orientedIndex];
accumulant += *pointer * coefficients[9];
accumulant2 += *pointer * weights[9];
pointer += 8 * cuePitchInFloats;
accumulant += *pointer * coefficients[10];
accumulant2 += *pointer * weights[10];
pointer += 8 * cuePitchInFloats;
accumulant += *pointer * coefficients[11];
accumulant2 += *pointer * weights[11];
devMpb[orientedIndex] = accumulant;
devCombinedg[orientedIndex] = accumulant2;
}
}
#include "includes.h"
__global__ void AccuracyDivideKernel(const int N, float* accuracy) {
*accuracy /= N;
}
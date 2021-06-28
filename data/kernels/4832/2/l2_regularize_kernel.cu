#include "includes.h"
__global__ void l2_regularize_kernel(int factors, float regularization, float * YtY) {
YtY[threadIdx.x * factors + threadIdx.x] += regularization;
}
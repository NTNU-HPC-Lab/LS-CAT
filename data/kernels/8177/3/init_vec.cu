#include "includes.h"

using namespace std;


// https://stackoverflow.com/questions/26853363/dot-product-for-dummies-with-cuda-c



__global__ void init_vec(float* vec, float value) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
vec[tid] = value;
}
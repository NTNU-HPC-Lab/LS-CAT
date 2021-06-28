#include "includes.h"
__device__ void get_largest_value(short *vec, const int vec_length, int *max) {

for (int i = threadIdx.x; i < vec_length; i = i + blockDim.x) {

atomicMax(max, vec[i]);
}
}
__global__ void get_largest_value(int *vec, const int vec_length, int* max) {

for (int i = threadIdx.x; i < vec_length; i = i + blockDim.x) {
atomicMax(max, vec[i]);
}

}
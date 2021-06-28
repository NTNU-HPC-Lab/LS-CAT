#include "includes.h"
__device__ void find_index(short *vec, const int vec_length, int *value, int *index) {
for (int i = threadIdx.x; i < vec_length; i = i + blockDim.x) {
if (vec[i] == *value) {
atomicMax(index, i);
}
}
}
__global__ void find_index(int *vec, const int vec_length, int *value, int *index){
for (int i = threadIdx.x; i < vec_length; i = i + blockDim.x) {
if(vec[i]==*value){
atomicMax(index, i);
}

}
}
#include "includes.h"
__global__ void kernelVector_suma_constante(float* array, int _size, int _constant){
int idx= blockIdx.x * blockDim.x + threadIdx.x;
if(idx < _size){
array[idx] = array[idx]+_constant;
}
}
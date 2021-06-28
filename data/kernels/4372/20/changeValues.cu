#include "includes.h"
__global__ void changeValues(float *matrix, int size) {
int index = (blockIdx.x * blockDim.x) + threadIdx.x;
if (index < size) {
float a = matrix[index] * 10;
int b = (int) a;
matrix[index] = (float) b;

}
}
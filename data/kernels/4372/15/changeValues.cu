#include "includes.h"
__global__ void changeValues(double *matrix, int size) {
int index = (blockIdx.x * blockDim.x) + threadIdx.x;
if (index < size) {
double a = matrix[index] * 10;
int b = (int) a;
matrix[index] = (double) b;

}
}
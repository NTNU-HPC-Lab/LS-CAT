#include "includes.h"
__global__ void kernel_1(int columns, int rows, float* mat1, float* matanswer) {
int columna = threadIdx.x;  //En que columna operamos (no filas)
float temp_value = 0;

for (int k = 0; k < rows; k++) {
temp_value = temp_value + mat1[(k * columns) + columna];
}
matanswer[columna] = temp_value;
}
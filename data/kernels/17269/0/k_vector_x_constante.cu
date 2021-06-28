#include "includes.h"

using namespace std;

void print_function(int*, int);

// Programando Tarjeta//


__global__ void k_vector_x_constante(int* arr, int* answer, int n, int k) { // arr -> Vector, n -> tamaño de array, k flotante
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < n) {
answer[idx] = arr[idx] * k;
}
}
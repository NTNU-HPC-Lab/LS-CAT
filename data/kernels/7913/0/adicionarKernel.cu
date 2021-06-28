#include "includes.h"



__global__ void adicionarKernel(double* resultado, const double* n) {
int i = threadIdx.x;
double a = 1, b = 0;
double delta = pow(b, 2) - (4 * a * (n[i] * -1));
resultado[i] = ((b * -1) + sqrt(delta)) / 2 * a;
}
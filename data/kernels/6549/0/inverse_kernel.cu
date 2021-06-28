#include "includes.h"
__global__ void inverse_kernel(double* d_y, double* d_x) {
double x = *d_x;
*d_y = 1. / x;
}
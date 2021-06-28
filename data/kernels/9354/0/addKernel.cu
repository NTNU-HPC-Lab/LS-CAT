#include "includes.h"

using namespace std;

// this amazingly nice error checking function is stolen from:
//https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
__global__ void addKernel(double *c, const double *a, const double *b) {
int i = threadIdx.x;
c[i] = a[i] + b[i];
}
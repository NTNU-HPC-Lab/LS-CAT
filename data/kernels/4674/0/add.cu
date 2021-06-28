#include "includes.h"
using namespace std;

// function generate random numbers and assign it to array
__global__ void add(int *a, int *b, int *c) {
int index = threadIdx.x + blockIdx.x * blockDim.x;
c[index] = a[index] + b[index];
}
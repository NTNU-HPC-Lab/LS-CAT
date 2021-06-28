#include "includes.h"
__global__ void multiply(int a, int b, int *c) {
*c = a * b;
}
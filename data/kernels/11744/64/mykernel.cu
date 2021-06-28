#include "includes.h"
__global__ void mykernel(int *int1, float *f1, int *int2) {
f1[0] = *(float *)&int1[0];
int2[0] = *(int *)&f1[0];
}
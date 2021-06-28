#include "includes.h"
__global__ void inttofloat(float *out, int *in) {
out[0] = *(float *)&in[0];
}
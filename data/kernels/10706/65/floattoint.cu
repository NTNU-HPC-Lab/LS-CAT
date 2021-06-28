#include "includes.h"
__global__ void floattoint(int *out, float *in) {
out[0] = *(int *)&in[0];
}
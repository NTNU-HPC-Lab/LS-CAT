#include "includes.h"
__global__ void apply_weights_kernel(double *g_out, int *g_in, double *g_ttmp) {
int val[2], test = 1;
double ttp_temp[2];
const int index = (blockIdx.x * blockDim.x + threadIdx.x) << 1;

val[0] = g_in[index];
val[1] = g_in[index + 1];
ttp_temp[0] = g_ttmp[index];
ttp_temp[1] = fabs(g_ttmp[index + 1]);

test = ttp_temp[0] < 0.0 ? 0 : 1;

g_out[index + 1] = (double) val[1] * ttp_temp[1];
ttp_temp[1] *= -g_ttp_inc[test];
g_out[index] = (double) val[0] * ttp_temp[1];
}
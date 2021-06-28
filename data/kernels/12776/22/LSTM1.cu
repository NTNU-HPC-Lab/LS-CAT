#include "includes.h"
__device__ float tanh_(float x)
{
// e**2x - 1
// ---------
// e**2x + 1
float exp2x =    exp(2.0*x);
return (exp2x - 1.0)/(exp2x + 1.0);
}
__global__ void LSTM1(float* layer1, float* lstm1, const float* gate1i, const float* gate1o, const int offset)
{
int i = blockDim.x*blockIdx.x + threadIdx.x; //256
float g_i = gate1i[256*offset + i];
float g_f = 1.0 - g_i;
float g_o = gate1o[256*offset + i];

float i_t = tanh_(layer1[256*offset + i]) * g_i;
float i_p = 0.0;
if (offset > 0)
i_p = g_f * lstm1[256*(offset-1) + i];
float sum = i_p + i_t;
lstm1[256*offset + i] = sum;
layer1[256*offset + i] = tanh_(sum) * g_o;
}
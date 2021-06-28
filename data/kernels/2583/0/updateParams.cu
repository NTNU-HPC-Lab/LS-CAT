#include "includes.h"

extern "C"
{










}
__global__ void updateParams(int N, int M, float alpha, float beta1, float beta2, float t, float *PARAMS, float *GRADS, float *m, float *v)
{

int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;

int index = j*N + i;

float beta1r = __fsub_rn(1.0, beta1);
float alphar = __fmul_rn(-alpha, __frcp_rn(__fsub_rn(1.0, __powf(beta1, t))));

if (i < N && j < M)
{
m[index] = __fmaf_rn(beta1, m[index], __fmul_rn(beta1r, GRADS[index]));
v[index] = fmaxf(fmaxf(__fmul_rn(beta2, v[index]), fabsf(GRADS[index])), 1.0e-16);
PARAMS[index] = __fmaf_rn(alphar,__fdividef(m[index], v[index]), PARAMS[index]);


//m[index] = beta1*m[index] + (1 - beta1)*GRADS[index];

//float a = beta2*v[index];
// float b = ((GRADS[index])>(0))?(GRADS[index]):(-GRADS[index]);
//float c = fmaxf(a, fabsf(GRADS[index])); // ((a)>(fabsf(GRADS[index]))?(a):(b);
//v[index] = fmaxf(c, 1.0e-16); // ((c)>(1.0e-16))?(c):(1.0e-16);
//float tmp = alpha/(1.0-powf(beta1, t));
//PARAMS[index] = PARAMS[index] - (alpha/(1.0-__powf(beta1, t)))*m[index]/v[index];
//PARAMS[index] = tmp*m[index]/v[index];
}
}
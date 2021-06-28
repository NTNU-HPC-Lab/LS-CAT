#include "includes.h"
__global__ void precalculateABC(float4* ABCm, float* M, float timestep, float alpha, unsigned int numPoints)
{
int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

if (me_idx>=numPoints)
return;

float twodelta = timestep*2.0f;
float deltasqr = timestep*timestep;


float Mii = M[me_idx];
float Dii = alpha*Mii;  // mass-proportional damping is applied

//	printf("M: %f\n",Mii);

float Ai = 1.0f/(Dii/twodelta + Mii/deltasqr);
float Bi = ((2.0f*Mii)/deltasqr)*Ai;
float Ci = (Dii/twodelta)*Ai - 0.5f*Bi;

//	printf("ABC for node %i: %f, %f, %f \n", me_idx, Ai, Bi, Ci);

ABCm[me_idx] = make_float4(Ai,Bi,Ci,Mii);
}
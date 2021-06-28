#include "includes.h"
__global__ void reg_GetConjugateGradient2_kernel(	float4 *nodeNMIGradientArray_d, float4 *conjugateG_d, float4 *conjugateH_d)
{
const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
if(tid < c_NodeNumber){
// G = - grad
float4 gradGValue = nodeNMIGradientArray_d[tid];
gradGValue = make_float4(-gradGValue.x, -gradGValue.y, -gradGValue.z, 0.0f);
conjugateG_d[tid]=gradGValue;

// H = G + gam * H
float4 gradHValue = conjugateH_d[tid];
gradHValue=make_float4(
gradGValue.x + c_ScalingFactor * gradHValue.x,
gradGValue.y + c_ScalingFactor * gradHValue.y,
gradGValue.z + c_ScalingFactor * gradHValue.z,
0.0f);
conjugateH_d[tid]=gradHValue;
nodeNMIGradientArray_d[tid]=make_float4(-gradHValue.x, -gradHValue.y, -gradHValue.z, 0.0f);
}
}
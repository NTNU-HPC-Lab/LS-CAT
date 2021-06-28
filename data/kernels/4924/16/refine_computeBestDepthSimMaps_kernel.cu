#include "includes.h"
__device__ float refineDepthSubPixel(const float3& depths, const float3& sims)
{
//    float floatDepth = depths.y;
float outDepth = -1.0f;

// subpixel refinement
// subpixel refine by Stereo Matching with Color-Weighted Correlation, Hierarchical Belief Propagation, and
// Occlusion Handling Qingxiong pami08
// quadratic polynomial interpolation is used to approximate the cost function between three discrete depth
// candidates: d, dA, and dB.
// TODO: get formula back from paper as it has been lost by encoding.
// d is the discrete depth with the minimal cost, dA ? d A 1, and dB ? d B 1. The cost function is approximated as f?x? ? ax2
// B bx B c.

float simM1 = sims.x;
float simP1 = sims.z;
float sim1 = sims.y;
simM1 = (simM1 + 1.0f) / 2.0f;
simP1 = (simP1 + 1.0f) / 2.0f;
sim1 = (sim1 + 1.0f) / 2.0f;

if((simM1 > sim1) && (simP1 > sim1))
{
float dispStep = -((simP1 - simM1) / (2.0f * (simP1 + simM1 - 2.0f * sim1)));

float floatDepthM1 = depths.x;
float floatDepthP1 = depths.z;

//-1 : floatDepthM1
// 0 : floatDepth
//+1 : floatDepthP1
// linear function fit
// f(x)=a*x+b
// floatDepthM1=-a+b
// floatDepthP1= a+b
// a = b - floatDepthM1
// floatDepthP1=2*b-floatDepthM1
float b = (floatDepthP1 + floatDepthM1) / 2.0f;
float a = b - floatDepthM1;

outDepth = a * dispStep + b;
};

return outDepth;
}
__global__ void refine_computeBestDepthSimMaps_kernel(float* osim, int osim_p, float* odpt, int odpt_p, float3* isims, int isims_p, float3* idpts, int idpts_p, int width, int height, float simThr)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

if((x < width) && (y < height))
{
float3 depths = idpts[y * idpts_p + x];
float3 sims = isims[y * isims_p + x];

float outDepth = ((sims.x < sims.y) ? depths.x : depths.y);
float outSim = ((sims.x < sims.y) ? sims.x : sims.y);
outDepth = ((sims.z < outSim) ? depths.z : outDepth);
outSim = ((sims.z < outSim) ? sims.z : outSim);

float refinedDepth = refineDepthSubPixel(depths, sims);
if(refinedDepth > 0.0f)
{
outDepth = refinedDepth;
};

osim[y * osim_p + x] = (outSim < simThr ? outSim : 1.0f);
odpt[y * odpt_p + x] = (outSim < simThr ? outDepth : -1.0f);
};
}
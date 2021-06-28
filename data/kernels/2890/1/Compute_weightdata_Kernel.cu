#include "includes.h"
__global__ void Compute_weightdata_Kernel(float* weightdata, const float* I, const float* input, int nPixels, int nChannels, int c, float norm_for_data_term, float eps)
{
int bx = blockIdx.x;
int tx = threadIdx.x;

int x = bx*blockDim.x + tx;
if (x >= nPixels)
return;

if (norm_for_data_term == 2)
{
weightdata[x] = 1;
}
else if (norm_for_data_term == 1)
{
weightdata[x] = 1.0f / (fabs(I[x] - input[x*nChannels + c]) + eps);
}
else
{
weightdata[x] = pow(fabs(I[x] - input[x*nChannels + c]) + eps, norm_for_data_term - 2);
}
}
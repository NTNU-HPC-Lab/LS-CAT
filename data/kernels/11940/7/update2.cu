#include "includes.h"
__global__ void update2(float *alphaMinusBeta_out, const float *rho, const float *yDotZ, const float *alpha)
{
const float beta = *rho * *yDotZ;
*alphaMinusBeta_out = *alpha - beta;
}
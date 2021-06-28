#include "includes.h"
__global__ void update3(float *rho_out, float *H0_out, const float *yDotS, const float *yDotY)
{
*rho_out = 1.0f / *yDotS;

if (*yDotY > 1e-5)
*H0_out = *yDotS / *yDotY;
}
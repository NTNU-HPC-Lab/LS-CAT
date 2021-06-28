#include "includes.h"
__global__ void createRaysOrthoKernel(float4* rays, int width, int height, float x0, float y0, float z, float dx, float dy, unsigned rayMask )
{
int rayx = threadIdx.x + blockIdx.x*blockDim.x;
int rayy = threadIdx.y + blockIdx.y*blockDim.y;
if( rayx >= width || rayy >= height )
return;

float tMinOrMask = 0.0f;
if( rayMask )
tMinOrMask = __int_as_float( rayMask );

int idx = rayx + rayy*width;
rays[2*idx+0] = make_float4( x0+rayx*dx, y0+rayy*dy, z, tMinOrMask );  // origin, tmin
rays[2*idx+1] = make_float4( 0, 0, 1, 1e34f ); // dir, tmax
}
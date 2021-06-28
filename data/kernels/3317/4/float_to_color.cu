#include "includes.h"
__device__ unsigned char value( float n1, float n2, int hue ) {
if (hue > 360)      hue -= 360;
else if (hue < 0)   hue += 360;

if (hue < 60)
return (unsigned char)(255 * (n1 + (n2-n1)*hue/60));
if (hue < 180)
return (unsigned char)(255 * n2);
if (hue < 240)
return (unsigned char)(255 * (n1 + (n2-n1)*(240-hue)/60));
return (unsigned char)(255 * n1);
}
__global__ void float_to_color( unsigned char *optr, const float *outSrc ) {
// map from threadIdx/BlockIdx to pixel position
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;
int offset = x + y * blockDim.x * gridDim.x;

float l = outSrc[offset];
float s = 1;
int h = (180 + (int)(360.0f * outSrc[offset])) % 360;
float m1, m2;

if (l <= 0.5f)
m2 = l * (1 + s);
else
m2 = l + s - l * s;
m1 = 2 * l - m2;

optr[offset*4 + 0] = value( m1, m2, h+120 );
optr[offset*4 + 1] = value( m1, m2, h );
optr[offset*4 + 2] = value( m1, m2, h -120 );
optr[offset*4 + 3] = 255;
}
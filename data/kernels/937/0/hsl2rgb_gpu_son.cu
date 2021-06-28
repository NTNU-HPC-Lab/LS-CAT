#include "includes.h"


__device__ float Hue_2_RGB_gpu( float v1, float v2, float vH )             //Function Hue_2_RGB
{
if ( vH < 0 ) vH += 1;
if ( vH > 1 ) vH -= 1;
if ( ( 6 * vH ) < 1 ) return ( v1 + ( v2 - v1 ) * 6 * vH );
if ( ( 2 * vH ) < 1 ) return ( v2 );
if ( ( 3 * vH ) < 2 ) return ( v1 + ( v2 - v1 ) * ( ( 2.0f/3.0f ) - vH ) * 6 );
return ( v1 );
}
__global__ void hsl2rgb_gpu_son(float * d_h , float * d_s ,unsigned char * d_l , unsigned char * d_r, unsigned char * d_g, unsigned char * d_b, int size)
{
int x = threadIdx.x + blockDim.x*blockIdx.x;
if (x >= size) return;
float H = d_h[x];
float S = d_s[x];
float L = d_l[x]/255.0f;
float var_1, var_2;
unsigned char r,g,b;

if ( S == 0 )
{
r = L * 255;
g = L * 255;
b = L * 255;
}
else
{

if ( L < 0.5 )
var_2 = L * ( 1 + S );
else
var_2 = ( L + S ) - ( S * L );

var_1 = 2 * L - var_2;
r = 255 * Hue_2_RGB_gpu( var_1, var_2, H + (1.0f/3.0f) );
g = 255 * Hue_2_RGB_gpu( var_1, var_2, H );
b = 255 * Hue_2_RGB_gpu( var_1, var_2, H - (1.0f/3.0f) );
}
d_r[x] = r;
d_g[x] = g;
d_b[x] = b;
}
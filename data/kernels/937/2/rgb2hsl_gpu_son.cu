#include "includes.h"


__global__ void rgb2hsl_gpu_son( unsigned char * d_r, unsigned char * d_g, unsigned char * d_b, float * d_h , float * d_s , unsigned char * d_l , int size)
{
int x = threadIdx.x + blockDim.x*blockIdx.x;
if (x >= size) return;

float H,S,L;
float var_r = ( (float)d_r[x]/255 );//Convert RGB to [0,1]
float var_g = ( (float)d_g[x]/255 );
float var_b = ( (float)d_b[x]/255 );
float var_min = (var_r < var_g) ? var_r : var_g;
var_min = (var_min < var_b) ? var_min : var_b;   //min. value of RGB
float var_max = (var_r > var_g) ? var_r : var_g;
var_max = (var_max > var_b) ? var_max : var_b;   //max. value of RGB
float del_max = var_max - var_min;               //Delta RGB value

L = ( var_max + var_min ) / 2;
if ( del_max == 0 )//This is a gray, no chroma...
{
H = 0;
S = 0;
}
else                                    //Chromatic data...
{
if ( L < 0.5 )
S = del_max/(var_max+var_min);
else {
S = del_max/(2-var_max-var_min );

float del_r = (((var_max-var_r)/6)+(del_max/2))/del_max;
float del_g = (((var_max-var_g)/6)+(del_max/2))/del_max;
float del_b = (((var_max-var_b)/6)+(del_max/2))/del_max;
if( var_r == var_max ) H = del_b - del_g;
else {
if( var_g == var_max ) H = (1.0/3.0) + del_r - del_b;
else H = (2.0/3.0) + del_g - del_r;
}
}

if ( H < 0 ) H += 1;
if ( H > 1 ) H -= 1;
}
d_h[x] = H;
d_s[x] = S;
d_l[x] = (unsigned char)(L*255);
}
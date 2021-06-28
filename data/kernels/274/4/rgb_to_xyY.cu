#include "includes.h"
__global__ void rgb_to_xyY( float* d_r, float* d_g, float* d_b, float* d_x, float* d_y, float* d_log_Y, float  delta, int    num_pixels_y, int    num_pixels_x )
{
int  ny             = num_pixels_y;
int  nx             = num_pixels_x;
int2 image_index_2d = make_int2( ( blockIdx.x * blockDim.x ) + threadIdx.x, ( blockIdx.y * blockDim.y ) + threadIdx.y );
int  image_index_1d = ( nx * image_index_2d.y ) + image_index_2d.x;

if ( image_index_2d.x < nx && image_index_2d.y < ny )
{
float r = d_r[ image_index_1d ];
float g = d_g[ image_index_1d ];
float b = d_b[ image_index_1d ];

float X = ( r * 0.4124f ) + ( g * 0.3576f ) + ( b * 0.1805f );
float Y = ( r * 0.2126f ) + ( g * 0.7152f ) + ( b * 0.0722f );
float Z = ( r * 0.0193f ) + ( g * 0.1192f ) + ( b * 0.9505f );

float L = X + Y + Z;
float x = X / L;
float y = Y / L;

float log_Y = log10f( delta + Y );

d_x[ image_index_1d ]     = x;
d_y[ image_index_1d ]     = y;
d_log_Y[ image_index_1d ] = log_Y;
}
}
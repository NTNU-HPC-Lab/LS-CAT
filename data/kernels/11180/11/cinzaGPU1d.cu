#include "includes.h"
__global__ void cinzaGPU1d( unsigned char *image1, unsigned char *res, int pixels ) {

int i = threadIdx.x + blockIdx.x*blockDim.x;
int cinza;

if( i < pixels ) {

int idx = 3*i;
int r = image1[ idx+2 ];
int g = image1[ idx+1 ];
int b = image1[ idx   ];

cinza  = (30*r + 59*g + 11*b)/100;

res[ idx+2 ] = (unsigned char)cinza;
res[ idx+1 ] = (unsigned char)cinza;
res[ idx   ] = (unsigned char)cinza;

}
}
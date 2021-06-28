#include "includes.h"
__global__ void mergeGPU1d( unsigned char *image1, unsigned char *image2, unsigned char *res, int pixels ) {

int i = threadIdx.x + blockIdx.x*blockDim.x;

if( i < pixels ) {

int idx = 3*i;
int r1 = image1[ idx+2 ];
int g1 = image1[ idx+1 ];
int b1 = image1[ idx   ];
int r2 = image2[ idx+2 ];
int g2 = image2[ idx+1 ];
int b2 = image2[ idx   ];
int r = (int)( ( (float)r1 + (float)r2 )*0.5f );
int g = (int)( ( (float)g1 + (float)g2 )*0.5f );
int b = (int)( ( (float)b1 + (float)b2 )*0.5f );
res[ idx+2 ] = (unsigned char)r;
res[ idx+1 ] = (unsigned char)g;
res[ idx   ] = (unsigned char)b;

}
}
#include "includes.h"
__global__ void initGuessBuffers( const uchar4* srcImg, float3* guess1, float3* guess2, const uint nRows, const uint nCols )
{
const uint nSamps = nRows*nCols;

const uint samp = threadIdx.x + blockDim.x * blockIdx.x;
if( samp < nSamps )
{
guess1[samp].x = srcImg[samp].x;
guess2[samp].x = srcImg[samp].x;

guess1[samp].y = srcImg[samp].y;
guess2[samp].y = srcImg[samp].y;

guess1[samp].z = srcImg[samp].z;
guess2[samp].z = srcImg[samp].z;
}
}
#include "includes.h"
/*
============================================================================
Name        :
Author      : Peter Whidden
Version     :
Copyright   :
Description :
============================================================================
*/




static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/*
* Device kernel that compares the provided PSF distribution to the distribution
* around each pixel in the provided image
*/


__global__ void convolvePSF(int width, int height, int imageCount, short *image, short *results, float *psf, int psfRad, int psfDim)
{
// Find bounds of image
const int x = blockIdx.x*32+threadIdx.x;
const int y = blockIdx.y*32+threadIdx.y;
const int minX = max(x-psfRad, 0);
const int minY = max(y-psfRad, 0);
const int maxX = min(x+psfRad, width);
const int maxY = min(y+psfRad, height);
const int dx = maxX-minX;
const int dy = maxY-minY;
if (dx < 1 || dy < 1) return;
// Read Image
/*__shared__*/ float convArea[13][13]; //convArea[dx][dy];
int xCorrection = x-psfRad < 0 ? 0 : psfDim-dx;
int yCorrection = y-psfRad < 0 ? 0 : psfDim-dy;
float sum = 0.0;
for (int i=0; i<dx; ++i)
{
for (int j=0; j<dy; ++j)
{
float value = float(image[0*width*height+(minX+i)*height+minY+j]);
sum += value;
convArea[i][j] = value;
}
}

float sumDifference = 0.0;
for (int i=0; i<dx; ++i)
{
for (int j=0; j<dy; ++j)
{
sumDifference += abs(convArea[i][j]/sum - psf[(i+xCorrection)*psfDim+j+yCorrection] );
}
}

results[0*width*height+x*height+y] = int(1000.0*sumDifference);//*/convArea[psfRad][psfRad]);

}
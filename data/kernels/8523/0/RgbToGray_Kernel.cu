#include "includes.h"
//==========================================================================
// Kernels
//==========================================================================
//==========================================================================
//==========================================================================
// End Kernels
//==========================================================================
//--------------------------------------------------------------------------
//==========================================================================
// Class Methods
//==========================================================================
__global__ void RgbToGray_Kernel(unsigned char * RGB_Image, unsigned char * Gray_Image, int Width, int Height)
{   //------------------------------------------------------------------
int globalX = blockIdx.x * blockDim.x + threadIdx.x;
int globalY = blockIdx.y * blockDim.y + threadIdx.y;
int OffsetGray = (globalY * Width + globalX);
int OffsetColor = (globalY * Width + globalX)*3;
//------------------------------------------------------------------

if(globalX<Width && globalY<Height)
{
Gray_Image[OffsetGray] = (unsigned char)(0.114f*RGB_Image[OffsetColor]+0.587f*RGB_Image[OffsetColor+1]+0.299f*RGB_Image[OffsetColor+2]);
}
}
#include "includes.h"
__global__ void copyPixelsInSlicesRGB(float *ptrinput0, float *ptrkslices0, int dH, int dW, int kH, int kW, int size1, int size2, int isize1, int isize2, int nInputPlane, int padleft, int padright, int padup, int paddown, int inputstr0, int kslicesstr0, int batchsize)
{
// each block does one pixel of the input image
// each kernel slice is represented by its upper-left coordinates

const int pixi=blockIdx.x;
const int pixj=blockIdx.y*blockDim.y + threadIdx.y;
const int tidx=threadIdx.x;
const int batchindex=blockIdx.z*blockDim.z+threadIdx.z;

int i,j;

int imin, jmin, imax, jmax;
int inputoffset, ksliceoffset;

// step 1 : find which kernel slices contain the values of the pixel
__shared__ int _imin, _jmin[32], _imax, _jmax[32], _inputoffset[32][3], _ksliceoffset[32][3];
if(threadIdx.z==0)
{
imin=(pixi - (kH - 1) + (dH -1))/dH > 0 ? (pixi - (kH - 1) + (dH -1))/dH : 0 ;
jmin=(pixj - (kW - 1) + (dW -1))/dW > 0 ? (pixj - (kW - 1) + (dW -1))/dW : 0 ;
imax= pixi / dH < size1 ? pixi / dH : size1 - 1 ;
jmax= pixj / dW < size2 ? pixj / dW : size2 - 1 ;
if(threadIdx.x==0 && threadIdx.y==0)
{
_imin=imin;
_imax=imax;
}
if(threadIdx.x==0)
{
_jmin[threadIdx.y]=jmin;
_jmax[threadIdx.y]=jmax;
}
inputoffset = inputstr0*blockIdx.z*blockDim.z + ((pixi-padup) * isize2 + (pixj-padleft)) * nInputPlane ;
ksliceoffset= kslicesstr0*blockIdx.z*blockDim.z + ((imin * size2  + jmin) * kH * kW +  (pixi - imin * dH) * kW + (pixj - jmin*dW) ) * nInputPlane;
_inputoffset[threadIdx.y][threadIdx.x]=inputoffset;
_ksliceoffset[threadIdx.y][threadIdx.x]=ksliceoffset;
}

__syncthreads();

if(batchindex >= batchsize) return;
if(pixj > isize2 + padleft + padright -1) return;


if(threadIdx.z>0)
{
imin=_imin;
imax=_imax;
jmin=_jmin[threadIdx.y];
jmax=_jmax[threadIdx.y];
inputoffset=_inputoffset[threadIdx.y][threadIdx.x];
ksliceoffset=_ksliceoffset[threadIdx.y][threadIdx.x];
}

// step 2 : move the pointers
// this one goes to where the pixel is at
ptrinput0   += inputoffset+inputstr0*threadIdx.z ;
ptrkslices0 += ksliceoffset+kslicesstr0*threadIdx.z ;

const int stridej = (kH*kW - dW) * nInputPlane;
const int stridei = (size2*kH-dH) * kW *nInputPlane - (jmax-jmin+1) * stridej ;

bool zeropad = pixi<padup || pixi>isize1-1+padup || pixj<padleft || pixj>isize2-1+padleft ;


// read pixel
// load the stuff first...
//for (b=0; b<batchsize; b++)
//{
float * ptrinput    = ptrinput0;
float * ptrkslices  = ptrkslices0;

float pixvalue;
if (zeropad) 	{
pixvalue=0;
}
else	{
pixvalue=ptrinput[tidx];
}


//	write to memory
for(i=imin; i<imax+1; i++) {
for(j=jmin; j<jmax+1; j++) {
if(zeropad)
{
ptrkslices[tidx]=0;
}
else {
ptrkslices[tidx]=pixvalue;
}
ptrkslices += stridej;
}
ptrkslices += stridei;
}
//}
}
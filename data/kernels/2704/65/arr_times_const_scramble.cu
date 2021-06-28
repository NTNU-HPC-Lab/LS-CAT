#include "includes.h"
__global__ void arr_times_const_scramble(float*a,float b, float * c, int sx,int sy,int sz, int ox, int oy, int oz)
{
int pnum=blockIdx.x*blockDim.x+threadIdx.x;   // which source array element do I have to deal with?

int px=pnum%(sx/2);   // my x pos of a complex number in the subarray
int py=pnum/(sx/2);   // my y pos of a complex number
if(px>=(sx/2) || py >= (sy/2)) return;  // not in range ... quit
int ids=2*(px+py*sx);  /// offset to array start in floats
int idd=2*((ox+px)+(oy+py)*sx);

// echange two values using a tmp
float tmpR = c[idd];
float tmpI = c[idd+1];
c[idd]=a[ids]; // (float)(ox+px); //
c[idd+1]=a[ids+1]; // (float)(oy+py); //
a[ids]=tmpR;
a[ids+1]=tmpI;
}
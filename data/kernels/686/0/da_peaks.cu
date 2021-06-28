#include "includes.h"
/*
* get_da_peaks is a gpu_accelerated local maxima finder
* [iprod] = get_da_peaks(i1, r, thresh);
* Written by Andrew Nelson 7/20/17
*
*
*
*
*/

// includes, project


// main
__global__ void da_peaks(float *d_i1, float thresh, int m, int n, int o)
{

int tx = threadIdx.x;
int ty = threadIdx.y;
float d_i2[25];
// location of output pixel being analyzed
int row_output = blockIdx.y*blockDim.y + ty;		// gives y coordinate as a function of tile width    **these lose meaning for (ty || tx) >= O_TILE_WIDTH and the same is true for **
int col_output = blockIdx.x*blockDim.x + tx;		// gives x coordinate as a function of tile width
int imnum = blockIdx.z;
if (imnum < o && row_output >=2 && row_output < m-2 && col_output >=2 && col_output <n-2)
{
// buffer the info into
for(int i = 0; i <5 ; i++){
for(int j = 0; j <5 ; j++)
{
d_i2[i*5 + j] = d_i1[(row_output - 2 + i) + (col_output - 2 +j)*m + imnum*m*n];
}
}
float me = d_i2[12];
int maxi = 1;
if(me < thresh){maxi = 0;}
for(int k = 0; k <25; k++)
{
if(d_i2[k] > me){maxi = 0;}
}
d_i1[row_output + col_output*m + imnum*m*n] = maxi;
}
else if(imnum <o){d_i1[row_output + col_output*m + imnum*m*n] = 0;}
else{}
}
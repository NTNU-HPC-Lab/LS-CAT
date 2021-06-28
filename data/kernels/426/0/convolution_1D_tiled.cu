#include "includes.h"


#define O_Tile_Width 3
#define Mask_width 3
#define width 5
#define Block_width (O_Tile_Width+(Mask_width-1))
#define Mask_radius (Mask_width/2)



__global__ void convolution_1D_tiled(float *N,float *M,float *P)
{
int index_out_x=blockIdx.x*O_Tile_Width+threadIdx.x;
int index_in_x=index_out_x-Mask_radius;
__shared__ float N_shared[Block_width];
float Pvalue=0.0;

//Load Data into shared Memory (into TILE)
if((index_in_x>=0)&&(index_in_x<width))
{
N_shared[threadIdx.x]=N[index_in_x];
}
else
{
N_shared[threadIdx.x]=0.0f;
}
__syncthreads();

//Calculate Convolution (Multiply TILE and Mask Arrays)
if(threadIdx.x<O_Tile_Width)
{
//Pvalue=0.0f;
for(int j=0;j<Mask_width;j++)
{
Pvalue+=M[j]*N_shared[j+threadIdx.x];
}
P[index_out_x]=Pvalue;
}


}
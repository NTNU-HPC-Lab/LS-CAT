#include "includes.h"
__global__ void histo_kernel_2 (unsigned char *buffer, int img_w, int img_h, int *histo)
{
int id_x = blockIdx.x * blockDim.x + threadIdx.x ;
int id_y = blockIdx.y * blockDim.y + threadIdx.y ;

atomicAdd (&histo[buffer[id_y*img_w + id_x]] , 1 );
}
#include "includes.h"
__global__ void kernel_push_atomic2( int *g_terminate, int *g_push_reser, int *s_push_reser, int *g_block_num, int width1)
{

int x  = __umul24( blockIdx.x, blockDim.x ) + threadIdx.x ;
int y  = __umul24( blockIdx.y , blockDim.y ) + threadIdx.y ;
int thid = __umul24( y , width1 ) + x ;

if( s_push_reser[thid] - g_push_reser[thid] != 0)
{
g_terminate[blockIdx.y * (*g_block_num) + blockIdx.x] = 1 ;
}

}
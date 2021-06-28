#include "includes.h"
__global__ void gpuSmMM( float *Ad , float *Bd , float *Cd , int dimention )
{

//Taking shared array to break the MAtrix in Tile widht and fatch them in that array per ele
__shared__ float Ads [tilewidth][tilewidth] ;
__shared__ float Bds [tilewidth][tilewidth] ;
// calculate thread id
unsigned int col = tilewidth*blockIdx.x + threadIdx.x ;
unsigned int row = tilewidth*blockIdx.y + threadIdx.y ;
for (int m = 0 ; m<dimention/tilewidth ; m++ ) // m indicate number of phase
{
Ads[threadIdx.y][threadIdx.x] =  Ad[row*dimention + (m*tilewidth + threadIdx.x)]  ;
Bds[threadIdx.y][threadIdx.x] =  Bd[ ( m*tilewidth + threadIdx.y) * dimention + col] ;
__syncthreads() ; // for syncronizeing the threads
// Do for tile
for ( int k1 = 0; k1<tilewidth ; k1++ )
Cd[row*dimention + col]+= Ads[threadIdx.x][k1] * Bds[k1][threadIdx.y] ;
__syncthreads() ; // for syncronizeing the threads

}
}
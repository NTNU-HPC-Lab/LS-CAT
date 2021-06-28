#include "includes.h"
__global__ void kernelMultMatTiled(float *d_M, float *d_N, float *d_P, int m,int n , int y){


// se define la memoria compartida de los tiles de tamaño TILE_WIDTH

__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

int bx = blockIdx.x;
int by = blockIdx.y;
int tx = threadIdx.x;
int ty = threadIdx.y;
int row = by * TILE_WIDTH + ty;
int col = bx * TILE_WIDTH + tx;
float Pvalue = 0;

for(int i = 0; i < n / TILE_WIDTH; i++){
/* primeramente se revisa que el elemento se encuentre en la matriz d_M ,
si no es así se establecen como cero
*/
if((i*TILE_WIDTH + tx) < n && row < m){
Mds[ty][tx]=d_M[row*n + (i*TILE_WIDTH + tx)];
}else{
Mds[ty][tx]=0.0;
}
/* despues  se revisa que el elemento se encuentre en la matriz d_N ,
si no es así se establecen como cero
*/
if((i*TILE_WIDTH + ty) < n && col < y){
Nds[ty][tx]= d_N[(i*TILE_WIDTH + ty)*y + col];
}else{
Nds[ty][tx]=0.0;
}
__syncthreads();
/*Se realiza la multiplicacion de elementos que están dentro del TILE
y se va guardando en Pvalue*/
for(int k = 0; k < TILE_WIDTH; ++k){
Pvalue += Mds[ty][k] * Nds[k][tx];
}
__syncthreads();
}
//se asigna el resultado de Pvalue en las posiciones de d_P
if(row<m && col < y)
d_P[(row*y)+ col] = Pvalue;
}
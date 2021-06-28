#include "includes.h"
__global__ void Evolve( int *val, int *aux, int n ) {
int up, upright, right, rightdown, down, downleft, left, leftup;
int sum = 0, estado;
const int tx = threadIdx.x + 1, ty = threadIdx.y + 1;
const int i = blockIdx.y * blockDim.y + threadIdx.y;
const int j = blockIdx.x * blockDim.x + threadIdx.x;
const int b2 = BSIZE + 2;
__shared__ float sdata[ b2 ][ b2 ];

sdata[ ty ][ tx ] = val[ i * n + j ];
if( ( tx == 1 ) && ( ty == 1 ) ) {
int stx = blockIdx.x * blockDim.x;
int sty = blockIdx.y * blockDim.y;
sdata[ 0      ][ 0      ] = val[ ( sty - 1     ) * n + stx - 1     ];
sdata[ 0      ][ b2 - 1 ] = val[ ( sty - 1     ) * n + stx + BSIZE ];
sdata[ b2 - 1 ][ 0      ] = val[ ( sty + BSIZE ) * n + stx - 1     ];
sdata[ b2 - 1 ][ b2 - 1 ] = val[ ( sty + BSIZE ) * n + stx + BSIZE ];
}
if( ( j > 0 ) && ( tx == 1 ) ) {
sdata[ ty     ][ 0      ] = val[ i * n + j - 1 ];
}
if( ( j < ( n - 1 ) ) && ( tx == BSIZE ) ) {
sdata[ ty     ][ b2 - 1 ] = val[ i * n + j + 1 ];
}
if( ( i > 0 ) && ( ty == 1 ) ) {
sdata[ 0      ][ tx     ] = val[ ( i - 1 ) * n + j ];
}
if( ( i < ( n - 1 ) ) && ( ty == BSIZE ) ) {
sdata[ b2 - 1 ][ tx     ] = val[ ( i + 1 ) * n + j ];
}
__syncthreads( );
if( ( i > 0 ) && ( i < ( n - 1 ) ) && ( j > 0 ) && ( j < ( n - 1 ) ) ) {
estado = sdata[ ty ][ tx ];
up = sdata[ ty - 1 ][ tx ];
upright = sdata[ ty - 1 ][ tx + 1 ];
right = sdata[ ty ][ tx + 1 ];
rightdown = sdata[ ty + 1 ][ tx + 1 ];
down = sdata[ ty + 1 ][ tx ];
downleft = sdata[ ty + 1 ][ tx - 1 ];
left = sdata[ ty ][ tx - 1 ];
leftup = sdata[ ty - 1 ][ tx - 1 ];
sum = up + upright + right + rightdown + down + downleft + left + leftup;
if( sum == 3 ) {
estado = 1;
}
else if( ( estado == 1 ) && ( ( sum < 2 ) || ( sum > 3 ) ) ) {
estado = 0;
}
aux[ i * n + j ] = estado;
}
}
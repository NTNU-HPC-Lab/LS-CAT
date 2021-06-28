#include "includes.h"
__global__ void Evolve( int *val, int *aux, int n ) {
int up, upright, right, rightdown, down, downleft, left, leftup;
int sum = 0, estado;
int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;
if ( i > 0 && i < (n - 1) && j > 0 && j < (n - 1) ){
estado = val[ i * n + j ];
up = val[ ( i - 1 ) * n + j ];
upright = val[ ( i - 1 ) * n + j + 1 ];
right = val[ i * n + j + 1 ];
rightdown = val[ ( i + 1 ) * n + j + 1 ];
down = val[ ( i + 1 ) * n + j ];
downleft = val[ ( i + 1 ) * n + j - 1 ];
left = val[ i * n + j - 1 ];
leftup = val[ ( i - 1 ) * n + j - 1 ];
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
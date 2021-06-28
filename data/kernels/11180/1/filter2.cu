#include "includes.h"
//-----------------------------------------
// Autor: Farias
// Data : January 2012
// Goal : Image treatment
//-----------------------------------------

/***************************************************************************************************
Includes
***************************************************************************************************/




/***************************************************************************************************
Defines
***************************************************************************************************/

#define ELEM(i,j,DIMX_) (i+(j)*(DIMX_))
#define BLOCK_SIZE 16


/***************************************************************************************************
Functions
***************************************************************************************************/

using namespace std;


/**************************************************************************************************/

__global__ void filter2( int width, int height, unsigned char *src, unsigned char *dest ) {

int i = threadIdx.x + blockIdx.x*blockDim.x;
int j = threadIdx.y + blockIdx.y*blockDim.y;

int aux, idx;

__shared__ int pesos[3][3];

// Setando Pesos
pesos[0][0] = 0; pesos[0][1] = 2; pesos[0][2] = 0;
pesos[1][0] = 2; pesos[1][1] = 4; pesos[1][2] = 2;
pesos[2][0] = 0; pesos[2][1] = 2; pesos[2][2] = 0;



if(i > 0 && j > 0 && i < width - 1 && j < height - 1) {
for (int k = 0; k < 3; ++k)
{

aux = 0;
for (int lin = 0; lin < 3; lin++)
{
for (int col = 0; col < 3; col++){
idx = 3*ELEM( i + lin - 1, j + col - 1, width );
aux += pesos[lin][col]*src[ idx+k ];
}
}
aux /= 12;
idx = 3*ELEM( i, j , width );
dest[ idx+k ] = (unsigned char)aux;

}

}
}
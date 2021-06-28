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

__global__ void filter1( int width, int height, unsigned char *src, unsigned char *dest ) {

int i = threadIdx.x + blockIdx.x*blockDim.x;
int j = threadIdx.y + blockIdx.y*blockDim.y;

int aux, idx;

if(i > 0 && j > 0 && i < width - 1 && j < height - 1) {
for (int k = 0; k < 3; ++k)
{
aux = 0;
idx = 3*ELEM( i, j, width );

aux += 4*src[ idx+k ];

idx = 3*ELEM( i-1, j, width );
aux+= 2*src[ idx+k ];

idx = 3*ELEM( i, j-1, width );
aux+= 2*src[ idx+k ];

idx = 3*ELEM( i+1, j, width );
aux+= 2*src[ idx+k ];

idx = 3*ELEM( i, j+1, width );
aux+= 2*src[ idx+k ];

aux /= 12;

idx = 3*ELEM( i, j, width );
dest[ idx+k ] = (unsigned char)aux;

}

}
}
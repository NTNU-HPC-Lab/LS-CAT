#include "includes.h"
/*
*  test.cu
*  Part of CTC/logistic_rng/
*
*  Created and authored by Yu-Hang Tang on 2015-03-20.
*  Copyright 2015. All rights reserved.
*
*  Users are NOT authorized
*  to employ the present software for their own publications
*  before getting a written permission from the author of this file.
*/




__global__ void generate( float *output, float trunk, int n_particle )
{
for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_particle; i += gridDim.x * blockDim.x ) {
for( int j = 0; j < n_particle; j++ ) {
output[ i + j * n_particle ] = logistic<11, float>( trunk, i, j );
}
}
}
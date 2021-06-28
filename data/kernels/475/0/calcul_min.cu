#include "includes.h"
__global__ void calcul_min( unsigned long *ord, int ind_start, int ind_end, unsigned long long *ymin, int *ind_min, int size_max_parallel ){

int a = threadIdx.x;
int size_tot = (ind_end - ind_start -1);

//On n'effectue pas le calcul aux indices ind_start ni ind_end
int nb_threads = ceilf((float)size_tot/(float)size_max_parallel);

//size of region to compute in the current thread
int size_parallel = ceilf( (float)size_tot/(float)nb_threads );


//have to be computed before the case of a different size_parallel value
int ind_start_loc = ind_start + a * size_parallel + 1;

if ( a == (nb_threads - 1) )
size_parallel = size_tot - (nb_threads - 1) * size_parallel;


unsigned long min_loc = ord[ind_start_loc];
int ind_min_loc = ind_start_loc;
int i = 0;

//printf("FINDING YMIN\n");

for ( i = ind_start_loc; i < ind_start_loc + size_parallel; i++ ){

//Looking for the lowest ordinate
if ( ord[i]< min_loc ){
min_loc = ord[i];
ind_min_loc = i;

}

}

atomicMin(ymin, min_loc);

__syncthreads();

if (*ymin == min_loc)
*ind_min = ind_min_loc;

return;
}
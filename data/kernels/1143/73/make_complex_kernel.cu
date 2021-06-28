#include "includes.h"
__device__ double2 make_complex(double in, int evolution_type){
double2 result;

switch(evolution_type){
// No change
case 0:
result.x = in;
result.y = 0;
break;
// Im. Time evolution
case 1:
result.x = exp(-in);
result.y = 0;
break;
// Real Time evolution
case 2:
result.x = cos(-in);
result.y = sin(-in);
break;
}

return result;
}
__global__ void make_complex_kernel(double *in, int *evolution_type, double2 *out){

//int id = threadIdx.x + blockIdx.x*blockDim.x;
//out[id] = make_complex(in[id], evolution_type[id]);
for (int i = 0; i < 3; ++i){
out[i] = make_complex(in[i], evolution_type[i]);
}
}
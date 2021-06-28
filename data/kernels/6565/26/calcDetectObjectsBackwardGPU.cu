#include "includes.h"
__device__ float activator_derivative( float x )
{
float sig = 1.0f / (1.0f + exp( -x ));
return sig * (1 - sig);
}
__global__ void calcDetectObjectsBackwardGPU( float *dz_in, float *dz, float *in, int batch_size, int in_size_x, int in_size_y, int in_size_z, int max_bounding_boxes, int max_classes )
{
int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

for( int i = 0; i < max_bounding_boxes; i=i+(4+max_classes)){
int index = id * (in_size_x * in_size_y * in_size_z) + i;

dz[index  ] = activator_derivative( in[index  ] ) * dz_in[index  ]; // x: sigmoid derivative * grads
dz[index+1] = activator_derivative( in[index+1] ) * dz_in[index+1]; // y: sigmoid derivative * grads
dz[index+2] = exp( in[index+2] ) * dz_in[index+2]; // w: exp * grads
dz[index+3] = exp( in[index+3] ) * dz_in[index+3]; // w: exp * grads
for( int c = 0; c <max_classes; ++c){
int index2 = id * (in_size_x * in_size_y * in_size_z) + i+4+c;
dz[index2] = activator_derivative( in[index2] ) * dz_in[index2]; // id: sigmoid derivative * grads
}
}

/* original code
for(int b = 0; b < dz_in.size.b; ++b ){
for( int i = 0; i < _max_bounding_boxes; i=i+(4+_max_classes)){
dz( b, i  , 0, 0 ) = activator_derivative( in( b, i  , 0, 0 ) ) * dz_in( b, i  , 0, 0 ); // x: sigmoid derivative * grads
dz( b, i+1, 0, 0 ) = activator_derivative( in( b, i+1 , 0, 0 ) ) * dz_in( b, i+1, 0, 0 ); // y: sigmoid derivative * grads
dz( b, i+2, 0, 0 ) = exp( in( b, i+2, 0, 0 ) ) * dz_in( b, i+2, 0, 0 ); // w: exp * grads
dz( b, i+3, 0, 0 ) = exp( in( b, i+3, 0, 0 ) ) * dz_in( b, i+3, 0, 0 ); // h: exp * grads
for( int c = 0; c <_max_classes; ++c){
dz( b, i+4+c, 0, 0 ) = activator_derivative( in( b, i+4+c , 0, 0 ) ) * dz_in( b, i+4+c , 0, 0 ); // id: sigmoid derivative * grads
}
}
}
*/
}
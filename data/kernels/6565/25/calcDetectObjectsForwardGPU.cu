#include "includes.h"
__global__ void calcDetectObjectsForwardGPU(float *in, float *out, int in_size_x, int in_size_y, int in_size_z, int max_bounding_boxes, int max_classes )
{
int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

for( int i = 0; i < max_bounding_boxes; i=i+(4+max_classes)){
int index = id * (in_size_x * in_size_y * in_size_z) + i;
out[index  ] = 1.0f / (1.0f + exp( -in[index  ] )); // x: sigmoid
out[index+1] = 1.0f / (1.0f + exp( -in[index+1] )); // y: sigmoid
out[index+2] = exp( in[index+2] ); // w: exp
out[index+3] = exp( in[index+3] ); // h: exp
for( int c = 0; c < max_classes; ++c){
int index2 = id * (in_size_x * in_size_y * in_size_z) + i+4+c;
out[index2] = 1.0f / (1.0f + exp( -in[index2] )); // id: sigmoid
}
}

/* original
for(int b = 0; b < in.size.b; ++b ){
for( int i = 0; i < _max_bounding_boxes; i=i+(4+_max_classes)){
out( b, i  , 0, 0 ) = 1.0f / (1.0f + exp( -in( b, i  , 0, 0 ) )); // x: sigmoid
out( b, i+1, 0, 0 ) = 1.0f / (1.0f + exp( -in( b, i+1, 0, 0 ) )); // y: sigmoid
out( b, i+2, 0, 0 ) = exp( in( b, i+2, 0, 0 ) ); // w: exp
out( b, i+3, 0, 0 ) = exp( in( b, i+3, 0, 0 ) ); // h: exp
for( int c = 0; c < _max_classes; ++c){
out( b, i+4+c, 0, 0 ) = 1.0f / (1.0f + exp( -in( b, i+4+c , 0, 0 ) )); // id: sigmoid
}
}
}
*/
}
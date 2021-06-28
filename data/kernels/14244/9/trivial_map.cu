#include "includes.h"
__global__ void trivial_map(int* inp_d, MyInt4* inp_lift, int inp_size) {
const unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
if(gid < inp_size) {
int el = inp_d[gid];
MyInt4 res(el,el,el,el);
if(el < 0) { res.x = 0;  res.y = 0;  res.z = 0; }
inp_lift[gid] = res;
}
}
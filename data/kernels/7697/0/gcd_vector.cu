#include "includes.h"

using namespace std;
int threads;


__global__ void gcd_vector(int * d_out, int integer_m){
int idx = threadIdx.x;
for(int i = idx; i<integer_m; i+=blockDim.x){
int u = i, v = integer_m;
while ( v != 0) {
int r = u % v;
u = v;
v = r;
}
if(u == 1){
d_out[idx]++;
}
}
}
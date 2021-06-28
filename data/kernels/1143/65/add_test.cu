#include "includes.h"
__device__ double2 add(double2 a, double2 b){
return {a.x+b.x, a.y+b.y};
}
__global__ void add_test(double2 *a, double2 *b, double2 *c){
c[0] = add(a[0],b[0]);
}
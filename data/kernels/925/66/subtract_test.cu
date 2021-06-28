#include "includes.h"
__device__ double2 subtract(double2 a, double2 b){
return {a.x-b.x, a.y-b.y};
}
__global__ void subtract_test(double2 *a, double2 *b, double2 *c){
c[0] = subtract(a[0],b[0]);
}
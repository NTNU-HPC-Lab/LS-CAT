#include "includes.h"
__device__ double2 pow(double2 a, int b){
double r = sqrt(a.x*a.x + a.y*a.y);
double theta = atan(a.y / a.x);
return{pow(r,b)*cos(b*theta),pow(r,b)*sin(b*theta)};
}
__global__ void pow_test(double2 *a, int b, double2 *c){
c[0] = pow(a[0],b);
}
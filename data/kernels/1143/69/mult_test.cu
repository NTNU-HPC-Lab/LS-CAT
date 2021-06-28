#include "includes.h"
__device__ double2 mult(double2 a, double2 b){
return {a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x};
}
__device__ double2 mult(double2 a, double b){
return {a.x*b, a.y*b};
}
__global__ void mult_test(double2 *a, double b, double2 *c){
c[0] = mult(a[0],b);
}
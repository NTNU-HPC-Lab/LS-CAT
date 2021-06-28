#include "includes.h"
__global__ void ecuaciones(int a, int b, int c, float *sol){

int index = threadIdx.x;
float d = 0;
float x=0, y=0;
d = b*b-4*a*c;
if (d > 0) {
x = (-b+sqrt(d))/(2*a);
y = (-b-sqrt(d))/(2*a);
sol[index] = x;
sol[index+1]=y;
}
else if (d == 0) {
x = (-b)/(2*a);
sol[index] = x;
}
}
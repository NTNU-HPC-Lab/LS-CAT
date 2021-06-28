#include "includes.h"
__global__ void Turtle(int *v1, int *v2, int *lead)
{
if (*v1 >= *v2)
{
printf("%d, %d, %d", -1, -1,-1);
}
else
{
printf("%d, %d, %d\n", *v1, *v2, *lead);
double _result = ((float)*lead)/(((float)*v2)-((float)*v1));
int h = _result;
int m = _result * 60 - h*60;
int s = (_result * 3600) -m*60;
printf("%.3f\n",_result);
printf("%d, %d, %d\n", h, m, s);

}
}
#include "includes.h"

//#define ITEM_COUNT 2
#define _PI 3.14159265358979323846
#define _PI2 1.57079632679489661923
#define _RAD 6372795





using namespace std;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);



__global__ void geo_invert(double2* d_dot1, double2* d_dot2, double* d_dist, double* d_azimut, long count)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < count)
{
d_dot1[idx].x = d_dot1[idx].x * _PI / 180;	//lat1
d_dot1[idx].y = d_dot1[idx].y * _PI / 180;	//lng1
d_dot2[idx].x = d_dot2[idx].x * _PI / 180;	//lat2
d_dot2[idx].y = d_dot2[idx].y * _PI / 180;	//lng2

double cl1, cl2, sl1, sl2, delta, cdelta, sdelta;
cl1 = cos(d_dot1[idx].x);
cl2 = cos(d_dot2[idx].x);
sl1 = sin(d_dot1[idx].x);
sl2 = sin(d_dot2[idx].x);
delta = d_dot2[idx].y - d_dot1[idx].y;
cdelta = cos(delta);
sdelta = sin(delta);

double x, y, z, ad, z2;
y = sqrt(pow(cl2*sdelta, 2) + pow(cl1*sl2 - sl1*cl2*cdelta, 2));
x = sl1*sl2 + cl1*cl2*cdelta;
ad = atan(y / x);
d_dist[idx] = ad * _RAD;

x = (cl1*sl2) - (sl1*cl2*cdelta);
y = sdelta*cl2;

if (x == 0)
{
if (y > 0)
z = -90;
else if (y < 0)
z = 90;
else if (y == 0)
z = 0;
}
else
{
z = atan(-y / x) * 180 / _PI;
if (x < 0)
{
z = z + 180;
}
}

z2 = z + 180.0f;

while (z2 >= 360)
{
z2 = z2 - 360;
}

z2 = z2 - 180;


z2 = -z2 * _PI / 180;
double anglerad2;
anglerad2 = z2 - ((2 * _PI) * floor(z2 / (2 * _PI)));
d_azimut[idx] = anglerad2 * 180 / _PI;


}
}
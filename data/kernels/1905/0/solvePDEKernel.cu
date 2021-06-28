#include "includes.h"
//using namespace Eigen;
using namespace std;

__device__ void setPhysicialParameters(float T, float *ce, float *pho, float *lamda)
{
float Ts = 1456.16f, Tl = 1522.69f, fs = 0.0f, L = 268000.0f;
if (T < Ts)
{
fs = 0;
*pho = 7250.0f;
*lamda = 50.0f;
*ce = 540.0f;
}

if (T >= Ts && T <= Tl)
{
fs = (Tl - T) / (Tl - Ts);
*pho = 7250.0f;
*lamda = fs * 25.0f + (1.0f - fs) * 50.0f;
*ce = 540.0f + L / (Tl - Ts);
}

if (T > Tl)
{
fs = 1;
*pho = 7250.0f;
*lamda = 28.0f;
*ce = 540.0f;
}
}
__device__ float setBoundaryCondition(int tstep, float tau, float Vcast, float *hPop, int Section, float *ccml)
{
float zposition = tstep * tau * fabs(Vcast);//ËÙ¶È³ËÒÔÊ±¼ä(Ê±¼äÍø¸ñ*Íø¸ñÊý£©,¸÷¸öÀäÈ´¶Î³¤¶È
float h = 0; //±íÃæ´«ÈÈÏµÊý

for (int i = 0; i < Section; i++)
{
if (zposition >= *(ccml + i) && zposition <= *(ccml + i + 1))//ÏÞ¶¨¸÷¸öÀäÈ´¶Î£¬Ã¿¸öÀäÈ´¶Î¶ÔÓ¦Ò»¸öh
{
h = *(hPop + blockIdx.x * Section + i);
}
}
return h;
}
__global__ void solvePDEKernel(float *hPop, float *T_Last, float *T_New, float *T_Surface, float Tw, float lamda, float pho, float ce, int ny, float dy, int nx, float dx, float tau, int tnpts, int tstep, float Vcast, int Section, float *ccml)
{
float ax, ay, T_Up, T_Down, T_Middle, T_Right, T_Left;
float h;
ax = tau * lamda / (pho * ce * dx * dx);
ay = tau * lamda / (pho * ce * dy * dy);

int i = threadIdx.x;
int j = threadIdx.y;
int tis = blockIdx.x * nx * ny + i * ny + j;
int L = ny;

setPhysicialParameters(T_Last[tis], &ce, &pho, &lamda);
h = setBoundaryCondition(tstep, tau, Vcast, hPop, Section, ccml);

if (i != 0 && i != (nx - 1) && j != 0 && j != (ny - 1))//ÖÐ¼ä
{
T_Right = T_Last[tis + L];
T_Left = T_Last[tis - L];
T_Middle = T_Last[tis];
T_Up = T_Last[tis + 1];
T_Down = T_Last[tis - 1];
T_New[tis] = ax * T_Right - (2 * ax + 2 * ay - 1) * T_Middle + ax * T_Left + ay * T_Up + ay * T_Down;
}
else if (i == 0 && j == 0)//µã1
{
T_Up = T_Last[tis + 1];
T_Middle = T_Last[tis];
T_Down = T_Last[tis + 1] - 2 * dx * h * (T_Last[tis] - Tw) / lamda;
T_Right = T_Last[tis + L];
T_Left = T_Last[tis + L] - 2 * dx * h * (T_Last[tis] - Tw) / lamda;
T_New[tis] = ax * T_Right - (2 * ax + 2 * ay - 1) * T_Middle + ax * T_Left + ay * T_Up + ay * T_Down;
}
else if (i == (nx - 1) && j == 0)//µã2
{
T_Up = T_Last[tis + 1];
T_Middle = T_Last[tis];
T_Down = T_Last[tis + 1] - 2 * dx * h * (T_Last[tis] - Tw) / lamda;
T_Left = T_Last[tis - L];
T_Right = T_Last[tis - L] - 2 * dx * h * (T_Last[tis] - Tw) / lamda;
T_New[tis] = ax * T_Right - (2 * ax + 2 * ay - 1) * T_Middle + ax * T_Left + ay * T_Up + ay * T_Down;
}
else if (i == 0 && j == (ny - 1))//µã3
{
T_Up = T_Last[tis - 1] - 2 * dx *h * (T_Last[tis] - Tw) / lamda;
T_Middle = T_Last[tis];
T_Down = T_Last[tis - 1];
T_Right = T_Last[tis + L];
T_Left = T_Last[tis + L] - 2 * dx *h * (T_Last[tis] - Tw) / lamda;
T_New[tis] = ax * T_Right - (2 * ax + 2 * ay - 1) * T_Middle + ax * T_Left + ay * T_Up + ay * T_Down;
}
else if (i == (nx - 1) && j == (ny - 1))//µã4
{
T_Up = T_Last[tis - 1] - 2 * dx *h * (T_Last[tis] - Tw) / lamda;
T_Middle = T_Last[tis];
T_Down = T_Last[tis - 1];
T_Right = T_Last[tis - L] - 2 * dx * h * (T_Last[tis] - Tw) / lamda;
T_Left = T_Last[tis - L];
T_New[tis] = ax * T_Right - (2 * ax + 2 * ay - 1) * T_Middle + ax * T_Left + ay * T_Up + ay * T_Down;
}
else if (i == 0 && j != 0 && j != (ny - 1))//±ß1
{
T_Up = T_Last[tis + 1];
T_Middle = T_Last[tis];
T_Down = T_Last[tis - 1];
T_Right = T_Last[tis + L];
T_Left = T_Last[tis + L] - 2 * dx * h * (T_Last[tis] - Tw) / lamda;
T_New[tis] = ax * T_Right - (2 * ax + 2 * ay - 1) * T_Middle + ax * T_Left + ay * T_Up + ay * T_Down;
}
else if (i == (nx - 1) && j != 0 && j != (ny - 1))//±ß2
{
T_Up = T_Last[tis + 1];
T_Middle = T_Last[tis];
T_Down = T_Last[tis - 1];
T_Right = T_Last[tis - L] - 2 * dx * h * (T_Last[tis] - Tw) / lamda;
T_Left = T_Last[tis - L];
T_New[tis] = ax * T_Right - (2 * ax + 2 * ay - 1) * T_Middle + ax * T_Left + ay * T_Up + ay * T_Down;
}
else if (i != 0 && i != (nx - 1) && j == 0)//±ß3
{
T_Up = T_Last[tis + 1];
T_Middle = T_Last[tis];
T_Down = T_Last[tis + 1] - 2 * dx * h* (T_Last[tis] - Tw) / lamda;
T_Right = T_Last[tis + L];
T_Left = T_Last[tis - L];
T_New[tis] = ax * T_Right - (2 * ax + 2 * ay - 1) * T_Middle + ax * T_Left + ay * T_Up + ay * T_Down;
}
else if (i != 0 && i != (nx - 1) && j == (ny - 1))//±ß4
{
T_Up = T_Last[tis - 1] - 2 * dx * h * (T_Last[tis] - Tw) / lamda;
T_Middle = T_Last[tis];
T_Down = T_Last[tis - 1];
T_Right = T_Last[tis + L];
T_Left = T_Last[tis - L];
T_New[tis] = ax * T_Right - (2 * ax + 2 * ay - 1) * T_Middle + ax * T_Left + ay * T_Up + ay * T_Down;
}
if (i == 0 && j == int((ny - 1)/2))
T_Surface[blockIdx.x * tnpts + tstep] = T_New[tis];
T_Last[tis] = T_New[tis];
__syncthreads();
}
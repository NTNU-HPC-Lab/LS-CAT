#include "includes.h"

#define PI  3.1415926535897932
#define MAXEQNS    10       // maximum number of differential equations in the system

const int itermax10 = 2;    // number of iterations to use for rk10
const int itermax12 = 1;    // number of additional iterations to use for rk12
const int neqns = 2;        // number of differential equations in the system
const double tol = 1.0e-10; // the error tolerance
const double tol10 = tol / 10;
const bool sho = true;      // set sho to true if you want the simple harmonic oscillator results
// set sho to false, if you want the predator - prey results

// the following constants are the 10th order method's coefficients
const double  a0 = 0;
__constant__ double  a1 = 0.11747233803526765;
__constant__ double  a2 = 0.35738424175967745;
__constant__ double  a3 = 0.64261575824032255;
__constant__ double  a4 = 0.88252766196473235;
const double  a5 = 1.0000000000000000;

__constant__ double  b10 = 0.047323231137709573;
__constant__ double  b11 = 0.077952072407795078;
__constant__ double  b12 = -0.010133421269900587;
__constant__ double  b13 = 0.0028864915990617097;
__constant__ double  b14 = -0.00055603583939812082;
__constant__ double  b20 = 0.021779075831486075;
__constant__ double  b21 = 0.22367959757928498;
__constant__ double  b22 = 0.12204792759220492;
__constant__ double  b23 = -0.012091266674498959;
__constant__ double  b24 = 0.0019689074312004371;
__constant__ double  b30 = 0.044887590835180592;
__constant__ double  b31 = 0.15973856856089786;
__constant__ double  b32 = 0.32285378852557547;
__constant__ double  b33 = 0.12204792759220492;
__constant__ double  b34 = -0.0069121172735362915;
__constant__ double  b40 = 0.019343435528957094;
__constant__ double  b41 = 0.22312684732165494;
__constant__ double  b42 = 0.23418268877986459;
__constant__ double  b43 = 0.32792261792646064;
__constant__ double  b44 = 0.077952072407795078;
const double  b50 = 0.066666666666666667;
const double  b51 = 0.10981508874708385;
const double  b52 = 0.37359383699761912;
const double  b53 = 0.18126454003786724;
const double  b54 = 0.26865986755076313;

const double  c0 = 0.033333333333333333;
const double  c1 = 0.18923747814892349;
const double  c2 = 0.27742918851774318;
const double  c3 = 0.27742918851774318;
const double  c4 = 0.18923747814892349;
const double  c5 = 0.033333333333333333;

// the following coefficients allow us to get rk12 internal xk values from rk10 fk values
__constant__ double  g10 = 0.043407276098971173;
__constant__ double  g11 = 0.049891561330903419;
__constant__ double  g12 = -0.012483721919363355;
__constant__ double  g13 = 0.0064848904066894701;
__constant__ double  g14 = -0.0038158693974615597;
__constant__ double  g15 = 0.0014039153409773882;
__constant__ double  g20 = 0.030385164419638569;
__constant__ double  g21 = 0.19605322645426044;
__constant__ double  g22 = 0.047860687574395354;
__constant__ double  g23 = -0.012887249003100515;
__constant__ double  g24 = 0.0064058521980400821;
__constant__ double  g25 = -0.0022420783785910372;
__constant__ double  g30 = 0.032291666666666667;
__constant__ double  g31 = 0.19311806292811784;
__constant__ double  g32 = 0.25797759963091718;
__constant__ double  g33 = 0.019451588886825999;
__constant__ double  g34 = -0.0038805847791943522;
__constant__ double  g35 = 0.0010416666666666667;
__constant__ double  g40 = 0.035575411711924371;
__constant__ double  g41 = 0.18283162595088341;
__constant__ double  g42 = 0.29031643752084369;
__constant__ double  g43 = 0.22956850094334782;
__constant__ double  g44 = -0.0068157483053369507;
__constant__ double  g45 = 0.0029481689136947641;
__constant__ double  g50 = 0.031929417992355945;
__constant__ double  g51 = 0.19305334754638505;
__constant__ double  g52 = 0.27094429811105371;
__constant__ double  g53 = 0.28991291043710653;
__constant__ double  g54 = 0.13934591681802007;
__constant__ double  g55 = -0.010073942765637839;
const double  g60 = 0.033333333333333333;
const double  g61 = 0.18923747814892349;
const double  g62 = 0.27742918851774318;
const double  g63 = 0.27742918851774318;
const double  g64 = 0.18923747814892349;
const double  g65 = 0.033333333333333333;

// the following constants are the 12th order method's coefficients
const double  ah0 = 0.0;
const double  ah1 = 0.084888051860716535;
const double  ah2 = 0.26557560326464289;
const double  ah3 = 0.50000000000000000;
const double  ah4 = 0.73442439673535711;
const double  ah5 = 0.91511194813928346;
const double  ah6 = 1.0000000000000000;

__constant__ double  bh10 = 0.033684534770907752;
__constant__ double  bh11 = 0.057301749935629582;
__constant__ double  bh12 = -0.0082444880936983822;
__constant__ double  bh13 = 0.0029151263642014432;
__constant__ double  bh14 = -0.00096482361331657787;
__constant__ double  bh15 = 0.00019595249699271744;
__constant__ double  bh20 = 0.015902242088596380;
__constant__ double  bh21 = 0.16276437062291593;
__constant__ double  bh22 = 0.096031583397703751;
__constant__ double  bh23 = -0.011758319711158930;
__constant__ double  bh24 = 0.0032543514515832418;
__constant__ double  bh25 = -0.00061862458499748489;
__constant__ double  bh30 = 0.031250000000000000;
__constant__ double  bh31 = 0.11881843285766042;
__constant__ double  bh32 = 0.24868761828096535;
__constant__ double  bh33 = 0.11000000000000000;
__constant__ double  bh34 = -0.010410996557394222;
__constant__ double  bh35 = 0.0016549454187684515;
__constant__ double  bh40 = 0.015902242088596380;
__constant__ double  bh41 = 0.15809680304274781;
__constant__ double  bh42 = 0.18880881534382426;
__constant__ double  bh43 = 0.28087114502765051;
__constant__ double  bh44 = 0.096031583397703751;
__constant__ double  bh45 = -0.0052861921651656089;
__constant__ double  bh50 = 0.033684534770907752;
__constant__ double  bh51 = 0.11440754737426645;
__constant__ double  bh52 = 0.24657204460460206;
__constant__ double  bh53 = 0.20929436236889375;
__constant__ double  bh54 = 0.25385170908498387;
__constant__ double  bh55 = 0.057301749935629582;
const double  bh60 = 0;
const double  bh61 = 0.19581988897471611;
const double  bh62 = 0.14418011102528389;
const double  bh63 = 0.32000000000000000;
const double  bh64 = 0.14418011102528389;
const double  bh65 = 0.19581988897471611;

const double  ch0 = 0.023809523809523810;
const double  ch1 = 0.13841302368078297;
const double  ch2 = 0.21587269060493131;
const double  ch3 = 0.24380952380952381;
const double  ch4 = 0.21587269060493131;
const double  ch5 = 0.13841302368078297;
const double  ch6 = 0.023809523809523810;

__global__ void Order10FkKernel(double*device_X_Total, double* device_X_Not, double* device_F_Not, double h, double*device_f)
{

int tx = threadIdx.x;
device_X_Total[tx] = device_X_Not[tx] + h*((g10*device_F_Not[tx])+ (g11 * device_f[tx]) + (g12 * device_f[tx+2])+ (g13 * device_f[tx + 4]) + (g14 * device_f[tx+ 6])+ (g15 *device_f[tx+8]));
__syncthreads();
device_X_Total[tx+2] = device_X_Not[tx] + h*((g20*device_F_Not[tx])+ (g21 * device_f[tx]) + (g22 * device_f[tx+2])+ (g23 * device_f[tx + 4]) + (g24 * device_f[tx+ 6])+ (g25 *device_f[tx+8]));
__syncthreads();
device_X_Total[tx+4] = device_X_Not[tx] + h*((g30*device_F_Not[tx])+ (g31 * device_f[tx]) + (g32 * device_f[tx+2])+ (g33 * device_f[tx + 4]) + (g34 * device_f[tx+ 6])+ (g35 *device_f[tx+8]));
__syncthreads();
device_X_Total[tx+6] = device_X_Not[tx] + h*((g40*device_F_Not[tx])+ (g41 * device_f[tx]) + (g42 * device_f[tx+2])+ (g43 * device_f[tx + 4]) + (g44 * device_f[tx+ 6])+ (g45 *device_f[tx+8]));
__syncthreads();
device_X_Total[tx+8] = device_X_Not[tx] + h*((g50*device_F_Not[tx])+ (g51 * device_f[tx]) + (g52 * device_f[tx+2])+ (g53 * device_f[tx + 4]) + (g54 * device_f[tx+ 6])+ (g55 *device_f[tx+8]));
__syncthreads();
}
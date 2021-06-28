#include "includes.h"
__global__ void exclscnmb2e(int *d_data0, int *d_output0, int *d_data1, int *d_output1, int *d_data2, int *d_output2, int *d_data3, int *d_output3, int *d_data4, int *d_output4, int *d_data5, int *d_output5, int *d_data6, int *d_output6, int *d_data7, int *d_output7) {
const int twid=threadIdx.x;
switch(blockIdx.x) {
case 0:
if(twid<2) {
d_output0[twid]=d_data0[0]*twid;
}
return;
case 1:
if(twid<2) {
d_output1[twid]=d_data1[0]*twid;
}
return;
case 2:
if(twid<2) {
d_output2[twid]=d_data2[0]*twid;
}
return;
case 3:
if(twid<2) {
d_output3[twid]=d_data3[0]*twid;
}
return;
case 4:
if(twid<2) {
d_output4[twid]=d_data4[0]*twid;
}
return;
case 5:
if(twid<2) {
d_output5[twid]=d_data5[0]*twid;
}
return;
case 6:
if(twid<2) {
d_output6[twid]=d_data6[0]*twid;
}
return;
case 7:
if(twid<2) {
d_output7[twid]=d_data7[0]*twid;
}
return;
}
}
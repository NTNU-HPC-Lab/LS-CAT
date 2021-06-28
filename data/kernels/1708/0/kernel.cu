#include "includes.h"

/****************************************************************************
* An experiment with cuda kernel invocation parameters. 2x3x4 threads on
* one block should yield 24 kernel invocations.
*
* Compile with:
*   nvcc -o cupass cupass.cu
*
* Dr Kevan Buckley, University of Wolverhampton, January 2018
*****************************************************************************/
__device__ int is_a_match(char *attempt){
char plain_password1[] ="AA1111";
char plain_password2[] ="AA1112";
char plain_password3[] ="AA1113";
char plain_password4[] ="AA1114";

char *q = attempt;
char *w = attempt;
char *e = attempt;
char *r = attempt;
char *pp1 = plain_password1;
char *pp2 = plain_password2;
char *pp3 = plain_password3;
char *pp4 = plain_password4;

while(*q ==*pp1){
if(*q == '\0')
{
printf("password:%s\n", plain_password1);
break;
}
q++;
pp1++;
}
while(*w ==*pp2){
if(*w == '\0')
{
printf("password:%s\n", plain_password2);
break;
}
w++;
pp2++;
}
while(*e ==*pp3){
if(*e == '\0')
{
printf("password:%s\n", plain_password3);
break;
}
e++;
pp3++;
}
while(*r ==*pp4){
if(*r == '\0')
{
printf("password: %s\n", plain_password4);
return 1;
}
r++;
pp4++;
}
return 0;
}
__global__ void kernel(){
char i1, i2, i3, i4;

char password[7];
password[6] ='\0';

int i = blockIdx.x +65;
int j = threadIdx.x+65;
char firstMatch =i;
char secondMatch =j;

password[0] =firstMatch;
password[1] =secondMatch;
for(i1='0'; i1<='9'; i1++){
for(i2='0'; i2<='9'; i2++){
for(i3='0'; i3<='9'; i3++){
for(i4='0'; i4<='9'; i4++){
password[2] =i1;
password[3] =i2;
password[4] =i3;
password[5] =i4;
if(is_a_match(password)){
}
else{
//printf("tried: %s\n",password);
}
}
}
}
}
}
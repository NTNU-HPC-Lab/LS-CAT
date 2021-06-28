#include "includes.h"
__device__ int is_a_match(char *attempt) {
char plain_password1[] = "BP9843";
char plain_password2[] = "RP6870";
char plain_password3[] = "AP6498";
char plain_password4[] = "IP1354";
char *m = attempt;
char *s = attempt;
char *c = attempt;
char *d = attempt;
char *p1 = plain_password1;
char *p2 = plain_password2;
char *p3 = plain_password3;
char *p4 = plain_password4;
while(*m == *p1) {
if(*m == '\0')
{
printf("Password: %s\n",plain_password1);
break;
}
m++;
p1++;
}while(*s == *p2) {
if(*s == '\0')
{
printf("Password: %s\n",plain_password2);
break;
}
s++;
p2++;
}
while(*c == *p3) {
if(*c == '\0')
{
printf("Password: %s\n",plain_password3);
break;
}
c++;
p3++;
}
while(*d == *p4) {
if(*d == '\0')
{
printf("Password: %s\n",plain_password4);
return 1;
}
d++;
p4++;
}
return 0;
}
__global__ void kernel() {
char i1,i2,i3,i4; //variables
char password[7];
password[6] = '\0';
//block id thread id initilized
int i = blockIdx.x+65;
int j = threadIdx.x+65;
char firstMatch = i;
char secondMatch = j;
password[0] = firstMatch;
password[1] = secondMatch;
for(i1='0'; i1<='9'; i1++){
for(i2='0'; i2<='9'; i2++){
for(i3='0'; i3<='9'; i3++){
for(i4='0'; i4<='9'; i4++){
password[2] = i1;
password[3] = i2;
password[4] = i3;
password[5] = i4;
if(is_a_match(password)) {
}
else {
//printf("tried: %s\n", password);
}}
}
}
}
}
#include "includes.h"


__device__ void showProgress(char* trystr, int tam, unsigned long long int id)
{
int n=0;

for(int i=0;i<tam;i++)
{
if('!' != trystr[i])
{
n=1;
break;
}
}

if(id % 1000000000 == 0) printf("\n----- Id = %lld\n", id);

if(n == 0) printf("\n***** %lld, Tam = %d -> ┤%s├\n", id, tam, trystr);
}
__device__ char * id2str(unsigned long long int n)
{
//Convert the id of the thread into the string in order to be checked
int b=96, r, asciChar, count=0;
char * a, * res;

a = (char*) malloc(100);

do
{
r=n%b;
asciChar=32+r;
a[count]=asciChar;
count++;
n=n/b;
}
while(n!=0);

res = (char*) malloc(count+1);

for(int i=count-1; i>=0; --i) res[count-i-1] = a[i];

free(a);

res[count] = '\0';

return res;
}
__device__ __host__  int shc(char* ca, int tam)
{
int h = 0;

for (int i = 0; i < tam; i++)
{
h = 31 * h + ca[i];
}

return h;
}
__global__ void findcollisions(int hash, unsigned long long int strLength)
{
unsigned long long int id = blockDim.x * blockIdx.x + threadIdx.x + strLength; //+ i* /*4e40*/;

char *trystr = id2str(id);
int hc;

int tam = 0;

while(trystr[tam]!='\0')
tam++;


//if (oldId < 50) printf("%lld, ┤%s├\n", id, trystr);
showProgress(trystr, tam, id);

hc = shc(trystr, tam);

if(hc == hash)
printf("Collision found for string ┤%s├. Hashcode %d\n", trystr, hc);

if(trystr[tam-1] == ' ')
{
char * reverse = (char *) malloc(tam+1);

for(int i=tam-1; i>=0; --i)
reverse[tam-i-1] = trystr[i];

reverse[tam] = '\0';

hc = shc(reverse, tam);

if(hc == hash)
printf("Collision found for string ┤%s├. Hashcode %d\n", reverse, hc);

free(reverse);
}
free(trystr);

}
/**
 * utils.h
 **/
 
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

double my_gettimeofday();
int read_param(char *name, unsigned long **data, int *n, int *l, int *h);
int read_param_cuda(char *name, unsigned long **data, int *n, int *l, unsigned long long *h);
int read_data(char *name, unsigned long **data, int n);


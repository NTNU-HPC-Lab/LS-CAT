#ifndef ISING_H
#define ISING_H

/* 
*******************************
** Sequential Implementation **
*******************************
*/
/*Initiate the process, iterate k times 
and break if there isn't any change*/
void ising(int *G, double *w, int k, int n);
/* 
*****************************
** Parallel Implementation **
*****************************
*/
void ising_parallel(int *G, double *w, int k, int n);

#endif
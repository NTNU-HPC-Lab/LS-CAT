#ifndef ising_h
#define ising_h

#include <stdint.h>
#define diff 1e-6f

//! Ising model evolution
/*!

  \param G      Spins on the square lattice             [n-by-n]
  \param w      Weight matrix                           [5-by-5]
  \param k      Number of iterations                    [scalar]
  \param n      Number of lattice points per dim        [scalar]

  NOTE: Both matrices G and w are stored in row-major format.
*/
void ising( int8_t *G, float *w, int k, int n);

#endif

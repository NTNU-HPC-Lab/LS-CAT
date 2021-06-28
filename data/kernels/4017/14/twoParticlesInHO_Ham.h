/*!
Two interacting particles in a harmonic oscillator Hamiltonian
*/
#ifndef _INCC2ParticlesHO
#define _INCC2ParticlesHO
#include "hamiltonian.h"
#include "mxutils.h"
#include "easyloggingcpp/easylogging++.h"


class C2ParticlesHO : public CHamiltonian {
  //! parameters for the HO hamiltonian [xmax, x0, omega, g/dx]
  double * d_params = NULL;
  //! Coordinate and momentum axes
  double * h_x = NULL;
  double * d_x = NULL;
  double * h_px = NULL;
  double * d_px = NULL;
  double dx = 0.0;
  double dpx = 0.0;
  size_t sz = 0;
  //! Pointers to device functions
  hamFunc * d_Pot = NULL;
  hamFunc * d_Inter = NULL;
  hamFunc * d_Nonl = NULL;
public:

  //! destructor
  ~C2ParticlesHO();

  //! Constructor
  /*!
  \param Nx : resolution
  \param xmax : x is in (-xmax,xmax]
  \param x0 : center of the harmonic oscillator
  \param omega : frequency of the HO
  \param g : discretized interactions strength = g
  */
  C2ParticlesHO(size_t Nx, double g, double xmax = 5.0, double x0 = 0.0, double omega = 1.0);

  //! Create momentum and coordinate vectors on both device and host
  int initializeVectors(size_t Nx, double xmax, double x0);

  void * getParams(){
    return this->d_params;
  }

  //! \return pointer to a device function V(x)
  hamFunc * timeDepPotential(void);

  /*!  |x1-x2|=d_r
  \return pointer to a device function U_int(|x1-x2|)
  */
  hamFunc * timeDepInteraction(void);

  /*!
  Time dep Non-linear part of the Hamiltonian prop to |Psi(x)|
  \return pointer to a device function V_nonlin(|Psi(x)|)
  */
  hamFunc * timeDepNonLin(void);

  /*!
    Define coordinate and momentum axes, and their resolution for future FFT steps
    NOTE: clean up with delete [] h_x; delete [] h_px
  */
  static int fftdef(size_t Nx, double xmax, double * &h_x, double * &h_px, double &dx, double &dpx);

  //!!Get coordinate vector on the host
  double * getHostX(){
    return this->h_x;
  }
  //! Get coordinate vector on the device
  double * getDeviceX(){
    return  this->d_x;
  }
  //! Get momentum vector host
  double * getHostMom(){
    return this->h_px;
  }
  //! Get momentum vector device
  double * getDeviceMom(){
    return this->d_px;
  }
  //! Get dx
  double getCoordStep(){
    return this->dx;
  }
  //! Get px
  double getMomStep(){
    return this->dpx;
  }
};
#endif

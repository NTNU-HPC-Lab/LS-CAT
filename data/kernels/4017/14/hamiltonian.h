#ifndef DEFCHAM
#define DEFCHAM
typedef double (*hamFunc)(const double, const double, const void *);
/*!
The Hamiltonian interface definition,
including time dependent potential
and the interaction functions as __device__ .
Every hamiltonian has to conform to this interface
in order to be used in the time evolution
*/
class CHamiltonian{
public:
  virtual void * getParams(void)=0;
  /*!
  \return pointer to a device function V(x, T)
  */
  virtual hamFunc * timeDepPotential(void)=0;

  /*!  |x1-x2|=d_r
  \return pointer to a device function U_int(|x1-x2|)
  */
  virtual hamFunc * timeDepInteraction(void)=0;

  /*!
  Time dep Non-linear part of the Hamiltonian prop to |Psi(x)|
  \return pointer to a device function V_nonlin(|Psi(x)|)
  */
  virtual hamFunc * timeDepNonLin(void)=0;

  //!!Get coordinate vector on the host
  virtual double * getHostX(void)=0;
  //! Get coordinate vector on the device
  virtual double * getDeviceX(void)=0;
  //! Get momentum vector host
  virtual double * getHostMom(void)=0;
  //! Get momentum vector device
  virtual double * getDeviceMom(void)=0;
  //! Get dx
  virtual double getCoordStep(void)=0;
  //! Get px
  virtual double getMomStep(void)=0;
};
#endif

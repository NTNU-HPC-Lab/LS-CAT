#ifndef ABSTRACTCNONLINEARPROBLEMJACOBIANHEADERDEF
#define ABSTRACTCNONLINEARPROBLEMJACOBIANHEADERDEF

#include <armadillo>

class AbstractNonlinearProblemJacobian
{

  public:

    virtual void ComputeDFDU( const arma::vec& u, arma::mat& dfdu) = 0;

};

#endif

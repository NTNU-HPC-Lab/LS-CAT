#ifndef ABSTRACTCNONLINEARPROBLEMHEADERDEF
#define ABSTRACTCNONLINEARPROBLEMHEADERDEF

#include <armadillo>

class AbstractNonlinearProblem
{

  public:

    virtual void ComputeF( const arma::vec& u, arma::vec& f) = 0;

    virtual void PostProcess() {};
};

#endif

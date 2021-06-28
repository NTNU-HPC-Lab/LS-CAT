#ifndef UNCERTAINTY_H
#define UNCERTAINTY_H

class ThreadManager;
typedef std::shared_ptr<ThreadManager> ThreadManagerPtr;

class Economic;
typedef std::shared_ptr<Economic> EconomicPtr;

class Optimiser;
typedef std::shared_ptr<Optimiser> OptimiserPtr;

class Uncertainty;
typedef std::shared_ptr<Uncertainty> UncertaintyPtr;

/**
 * Class for managing %Uncertainty objects
 *
 * Uncertainties are modelled as Mean Reverting (Ornstein-Uhlenbeck processes)
 * with jumps. The probability distribution for jumps is assumed independent of
 * the mean reversion model.
 *
 * This model also includes diffusion.
 *
 * This will later include the ability to have non-stationary long-term means.
 *
 * See: http://marcoagd.usuarios.rdc.puc-rio.br/sim_stoc_proc.html
 * Langrenet et al. 2015
 */
class Uncertainty : public std::enable_shared_from_this<Uncertainty> {

public:
    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    /**
     * Constructor I
     *
     * Constructs an %Uncertainty object with default values
     */
    Uncertainty(OptimiserPtr optimiser);

    /**
     * Constructor II
     *
     * Constructs an %Uncertainty object with assigned values
     */
    Uncertainty(OptimiserPtr optimiser, std::string nm, double curr, double mp,
            double sd, double rev, double pj, double jp, bool active);

    /**
     * Destructor
     */
    ~Uncertainty();

    // ACCESSORS //////////////////////////////////////////////////////////////

    /**
     * Returns the Optimiser
     *
     * @return Optimiser as OptimiserPtr
     */
    OptimiserPtr getOptimiser() {
        return this->optimiser.lock();
    }
    /**
     * Sets the Optimiser
     *
     * @param opt as OptimiserPtr
     */
    void setOptimiser(OptimiserPtr opt) {
        this->optimiser = opt;
    }

    /**
     * Returns the name
     *
     * @return Name as std::string
     */
    std::string getName() {
        return this->name;
    }
    /**
     * Sets the name
     *
     * @param nm as std::string
     */
    void setName(std::string nm) {
        this->name = nm;
    }

    /**
     * Returns the current level of the uncertainty
     *
     * @return Current as double
     */
    double getCurrent() {
        return this->current;
    }
    /**
     * Sets the current level of the uncertainty
     *
     * @param curr as double
     */
    void setCurrent(double curr) {
        this->current = curr;
    }

    /**
     * Returns the long run mean
     *
     * @return Long run mean as double
     */
    double getMean() {
        return this->meanP;
    }
    /**
     * Sets the long run mean
     *
     * @param mean as double
     */
    void setMean(double mean) {
        this->meanP = mean;
    }

    /**
     * Returns standard deviation for noise
     *
     * @return Standard deviation as double
     * @note This process contains diffusion so this standard deviation is
     * multiplied by the prevailing level of the uncertainty.
     */
    double getNoiseSD() {
        return this->standardDev;
    }
    /**
     * Sets the standard deviation for noise
     *
     * @param sd as double
     */
    void setNoiseSD(double sd) {
        this->standardDev = sd;
    }

    /**
     * Returns the strength of mean reversion
     *
     * @return Mean reversion strength as double
     */
    double getMRStrength() {
        return this->reversion;
    }
    /**
     * Sets the strength of mean reversion
     *
     * @param mrs as double
     */
    void setMRStrength(double mrs) {
        this->reversion = mrs;
    }

    /**
     * Returns whether the commodity is active
     *
     * @return Active status as bool
     */
    bool getStatus() {
        return this->active;
    }
    /**
     * Sets whether the commodity is active
     *
     * @param status as bool
     */
    void setStatus(bool status) {
        this->active = status;
    }

    /**
     * Returns the expected present value of future cash flows of one unit use
     *
     * @return Present value as double
     */
    double getExpPV() {
        return this->expPV;
    }
    /**
     * Sets the expected present value of future cash flows of one unit use
     *
     * @param pv as double
     */
    void setExpPV(double pv) {
        this->expPV = pv;
    }

    /**
     * Returns the SD of present value of future cash flows of one unit use
     *
     * @return Present value SD as double
     */
    double getExpPVSD() {
        return this->expPVSD;
    }
    /**
     * Sets the SD of present value of future cash flows of one unit use
     *
     * @param pvsd as double
     */
    void setExpPVSD(double pvsd) {
        this->expPVSD = pvsd;
    }

    /**
     * Returns the exponential distribution parameter for jump size
     *
     * @return Jump size distribution parameter as double
     */
    double getPoissonJump() {
        return this->poissonJump;
    }
    /**
     * Sets the exponential distribution parameter for jump sizeof
     *
     * @param p as double
     */
    void setPoissonJump(double p) {
        this->poissonJump = p;
    }
    /**
     * Returns the probability of a jump at each time step
     *
     * @return Jump probability as double
     */
    double getJumpProb() {
        return this->jumpProb;
    }
    /**
     * Sets the probability of a jump at each time step
     *
     * @param prob as double
     */
    void setJumpProb(double prob) {
        this->jumpProb = prob;
    }

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

    /**
     * Runs Monte Carlo simulation to determine the expected present value
     *
     * This function uses Monte Carlo simulation to compute possible paths for
     * the unit price to take over time. These paths are discounted and the
     * average is taken to determine the present value of a constant one unit
     * of usage for the design horizon. This function calls std::thread to run
     * the simulation values in parallel.
     *
     * This is used to find the expected present value of a financial cost
     * such as fuel price if it is used the same amount for every period on the
     * horizon.
     *
     * @note This function is called only once for a road optimisation and is
     * therefore not computed for every road.
     */
    void computeExpPV();

private:
    std::weak_ptr<Optimiser> optimiser; /**< Weak pointer to Optimiser object*/
    std::string name;                   /**< Name of the product */
    double current;                     /**< Current level of uncertainty */
    double meanP;                       /**< Long-run mean */
    double trend;                       /**< Linear increase/decrease */
    double standardDev;                 /**< Standard deviation */
    double reversion;                   /**< Strength of mean reversion */
    double poissonJump;                 /**< Poisson parameter for jump sizes */
    double jumpProb;                    /**< Probability of a jump at each time */
    double expPV;                       /**< Expected PV of all future CF from MC sims per unit */
    double expPVSD;                     /**< Standard deviation of PV of all future CF from MC sims */
    bool active;                        /**< Used in problem? */

    // Private functions //////////////////////////////////////////////////////
    double singlePathValue();
    UncertaintyPtr me();          /**< Enables sharing from here */
};

#endif

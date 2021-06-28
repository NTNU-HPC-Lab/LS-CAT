#ifndef EARTHWORKCOSTS_H
#define EARTHWORKCOSTS_H

class EarthworkCosts;
typedef std::shared_ptr<EarthworkCosts> EarthworkCostsPtr;

/**
 * Class for managing earthwork costs
 */
class EarthworkCosts : public std::enable_shared_from_this<EarthworkCosts> {

public:
	// CONSTRUCTORS AND DESTRUCTORS //////////////////////////////////////////////
	/**
	 * Constructor
	 *
	 * Constructs an %Earthwork object with default values.
	 */
    EarthworkCosts(const Eigen::VectorXd &cd, const Eigen::VectorXd &cc,
            double fc);

	/**
	 * Destructor
	 */
	~EarthworkCosts();
	
	// ACCESSORS /////////////////////////////////////////////////////////////////
	/**
	 * Returns the vector of cut depth thresholds
	 *
     * @return Cut depths as const Eigen::VectorXd&
	 */
    const Eigen::VectorXd& getDepths() {
        return this->cDepths;
	}
	/**
	 * Sets the vector of cut depth thresholds
	 *
     * @param cd as const Eigen::VectorXd&
	 */
    void setDepths(const Eigen::VectorXd& cd) {
        this->cCosts = cd;
	}

	/**
	 * Returns the vector of corresponding cut costs
	 *
     * @return Cut costs as const Eigen::VectorXd&
	 */
    const Eigen::VectorXd& getCutCosts() {
        return this->cCosts;
	}
	/**
	 * Sets the vector of corresponding cut costs
	 *
     * @param cc as const Eigen::VectorXd&
	 */
    void setCutCosts(const Eigen::VectorXd& cc) {
        this->cCosts = cc;
	}

	/**
	 * Returns the fill cost
	 *
	 * @return Fill cost as double
	 */
	double getFillCost() {
		return this->fCost;
	}
	/**
	 * Sets the fill cost
	 *
	 * @param Fill cost as double
	 */
	void setFillCost(double fc) {
		this->fCost = fc;
	}

	// STATIC ROUTINES ///////////////////////////////////////////////////////////

	// CALCULATION ROUTINES //////////////////////////////////////////////////////

private:
        Eigen::VectorXd cDepths;    /**< Vector of cut depths */
        Eigen::VectorXd cCosts;	    /**< Corresponding vector of cut costs */
	double fCost;		    /**< Fill cost per m^3 */
};

#endif

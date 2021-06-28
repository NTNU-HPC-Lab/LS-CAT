#ifndef POLICYMAPFRONTIER_H
#define POLICYMAPFRONTIER_H

class PolicyMapFrontier;
typedef std::shared_ptr<PolicyMapFrontier> PolicyMapFrontierPtr;

/**
 * Class for managing policy map frontiers
 */
class PolicyMapFrontier : public std::enable_shared_from_this<PolicyMapFrontier> {

public:
    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    /**
     * Constructor I
     *
     * Constructs an empty %PolicyMapFrontier object
     */
    PolicyMapFrontier();

    /**
     * Constructor II
     *
     * Constructs a %PolicyMapFrontier object based and proposed options
     */
    PolicyMapFrontier(unsigned long base, unsigned long proposed);

    /**
     * Constructor II
     *
     * Constructs a %PolicyMapFrontier object with all attributes
     */
    PolicyMapFrontier(unsigned long base, unsigned long proposed,
        const Eigen::MatrixXd& lvls,
        const Eigen::VectorXd& unitProfit);

    /**
     * Destructor
     */
    ~PolicyMapFrontier();

    // ACCESSORS //////////////////////////////////////////////////////////////

    /**
     * Returns the base option
     *
     * @return Base option as unsigned long
     */
    unsigned long getBase() {
            return this->baseOption;
    }
    /**
     * Sets the base option
     *
     * @param bo as unsigned long
     */
    void setBase(unsigned long bo) {
            this->baseOption = bo;
    }

    /**
     * Returns the proposed option
     *
     * @return Proposed option as unsigned long
     */
    unsigned long getProposed() {
            return this->proposedOption;
    }
    /**
     * Sets the proposed option
     *
     * @param po as unsigned long
     */
    void setProposed(unsigned long po) {
            this->proposedOption = po;
    }

    /**
     * Returns the state levels
     *
     * @return State levels as const Eigen::MatrixXd&
     */
    const Eigen::MatrixXd& getStateLevels() {
        return this->stateLevels;
    }
    /**
     * Sets the state levels
     *
     * @param lvls as const Eigen::MatrixXd&
     */
    void setStateLevels(const Eigen::MatrixXd& lvls) {
        this->stateLevels = lvls;
    }

    /**
     * Returns the corresponding unit profits on the frontier
     *
     * @return Frontier unit profits as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getProfits() {
        return this->unitProfit;
    }
    /**
     * Sets the corresponding unit profits on the frontier
     *
     * @param profs as const Eigen::VectorXd&
     */
    void setProfits(const Eigen::VectorXd& profs) {
        this->unitProfit = profs;
    }

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

private:
    unsigned long baseOption;       /**< Prevailing control */
    unsigned long proposedOption;   /**< Proposed control */
    Eigen::MatrixXd stateLevels;    /**< Populations for frontier */
    Eigen::VectorXd unitProfit;     /**< Unit profit along frontier */
};

#endif

#ifndef POLICYMAPYEAR_H
#define POLICYMAPYEAR_H

class PolicyMapFrontier;
typedef std::shared_ptr<PolicyMapFrontier> PolicyMapFrontierPtr;

class PolicyMapYear;
typedef std::shared_ptr<PolicyMapYear> PolicyMapYearPtr;

/**
 * Class for managing the years for a policy map
 */
class PolicyMapYear : public std::enable_shared_from_this<PolicyMapYear> {

public:
    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    /**
     * Constructor I
     *
     * Constructs an empty %PolicyMapYear object
     */
    PolicyMapYear(unsigned long noDims, unsigned long noPaths);

    /**
     * Constructor II
     *
     * Constructs a %PolicyMapYear object by passing values
     */
    PolicyMapYear(const std::vector<PolicyMapFrontierPtr>& frontiers,
            const Eigen::MatrixXd& stateLevels,
            const Eigen::VectorXd& expectedProfit,
            const Eigen::VectorXi& optimalControls);

    /**
     * Destructor
     */
    ~PolicyMapYear();

    // ACCESSORS //////////////////////////////////////////////////////////////

    /**
     * Returns the frontiers
     *
     * @return Frontiers as const std::vector<PolicyMapFrontierPtr>&
     */
    const std::vector<PolicyMapFrontierPtr>& getFrontiers() {
        return this->frontiers;
    }
    /**
     * Sets the frontiers
     *
     * @param front as const std::vector<PolicyMapFrontierPtr>&
     */
    void setFrontiers(const std::vector<PolicyMapFrontierPtr>& front) {
        this->frontiers = front;
    }

    /**
     * Returns the state levels for data points
     *
     * @return State levels as const Eigen::MatrixXd&
     */
    const Eigen::MatrixXd& getStateLevels() {
        return this->stateLevels;
    }
    /**
     * Sets the state levels for data points
     *
     * @param pops as const Eigen::MatrixXd&
     */
    void setStateLevels(const Eigen::MatrixXd& sl) {
        this->stateLevels = sl;
    }

    /**
     * Returns the expected profits for data points
     *
     * @return Expected profits as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getProfits() {
        return this->expectedProfit;
    }
    /**
     * Sets the expected profits for data points
     *
     * @param profs as std::vector<double>*
     */
    void setProfits(const Eigen::VectorXd& profs) {
        this->expectedProfit = profs;
    }

    /**
     * Returns the corresponding optimal controls for each path
     *
     * @return Optimal controls vector as Eigen::VectorXi&
     */
    const Eigen::VectorXi& getOptConts() {
        return this->optimalControl;
    }
    /**
     * Sets the corresponding optimal controls for each path
     *
     * @param oc as Eigen::VectorXi&
     */
    void setOptConts(const Eigen::VectorXi& oc) {
        this->optimalControl = oc;
    }

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

private:
    std::vector<PolicyMapFrontierPtr> frontiers;    /**< Frontiers produced */
    Eigen::MatrixXd stateLevels;                    /**< State values */
    Eigen::VectorXd expectedProfit;                 /**< Corresponding E(Profit) */
    Eigen::VectorXi optimalControl;                 /**< Corresponding Optimal Control */
};

#endif

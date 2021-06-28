#ifndef PROGRAM_H
#define PROGRAM_H

class Program;
typedef std::shared_ptr<Program> ProgramPtr;

/**
 * Class for managing policy programs for ROV.
 */
class Program : public std::enable_shared_from_this<Program> {

public:
    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    /**
     * Constructor Constructs a %Program object with default values.
     */
    Program(const Eigen::VectorXd& flowRates, const Eigen::MatrixXd&
            switching);

    /**
     * Destructor
     */
    ~Program();

    // ACCESSORS //////////////////////////////////////////////////////////////

    /**
     * Returns the vector of flow rates
     *
     * @return Flow rates as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getFlowRates() {
        return this->flowRates;
    }
    /**
     * Sets the vector of flow rates
     *
     * @param rates as const Eigen::VectorXd&
     */
    void setFlowRates(const Eigen::VectorXd& rates) {
        this->flowRates = rates;
    }

    /**
     * Returns the vector of flow values
     *
     * @return Flow values as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getFlowValues() {
        return this->flowValues;
    }
    /**
     * Sets the vector of flow values
     *
     * @param values as const Eigen::VectorXd&
     */
    void setFlowValues(const Eigen::VectorXd& values) {
        this->flowValues = values;
    }

    /**
     * Returns the matrix of switching costs
     *
     * @return Switching matrix as const Eigen::MatrixXd&
     */
    const Eigen::MatrixXd& getSwitchingCosts() {
        return this->switching;
    }
    /**
     * Sets the matrix of switching costs
     *
     * @param costs as const Eigen::MatrixXd&
     */
    void setSwitchingCosts(const Eigen::MatrixXd& costs) {
        this->switching = costs;
    }

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

private:
    unsigned long number;		/**< Program identifier */
    Eigen::VectorXd flowRates;		/**< Flow rate options associated with program */
    Eigen::VectorXd flowValues;         /**< Corresponding values of each flow rate */
    Eigen::MatrixXd switching;		/**< Switching costs between controls */
};

#endif

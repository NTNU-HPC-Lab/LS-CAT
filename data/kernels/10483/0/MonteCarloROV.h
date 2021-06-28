#ifndef MONTECARLOROV_H
#define MONTECARLOROV_H

class PolicyMap;
typedef std::shared_ptr<PolicyMap> PolicyMapPtr;

class State;
typedef std::shared_ptr<State> StatePtr;

class Road;
typedef std::shared_ptr<Road> RoadPtr;

class MonteCarloROV;
typedef std::shared_ptr<MonteCarloROV> MonteCarloROVPtr;

/**
 * Class for managing ROV analysis with Monte Carlo simulation
 */
class MonteCarloROV {

public:
    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    /**
     * Constructor I
     *
     * Constructs an empty ROV object
     */
    MonteCarloROV();

    /**
     * Destructor
     */
    ~MonteCarloROV();

    // ACCESSORS //////////////////////////////////////////////////////////////

    /**
     * Returns the State
     *
     * @return State as StatePtr
     */
    StatePtr getState() {
        return this->state;
    }
    /**
     * Sets the State
     *
     * @param state as StatePtr
     */
    void setState(StatePtr state) {
        this->state.reset();
        this->state = state;
    }

    /**
     * Returns the ROV policy map
     *
     * @return PolicyMap as PolicyMapPtr
     */
    PolicyMapPtr getPolicyMap() {
        return this->policyMap;
    }
    /**
     * Sets the ROV policy map
     *
     * @param pm as PolicyMapPtr
     */
    void setPolicyMap(PolicyMapPtr pm) {
        this->policyMap.reset();
        this->policyMap = pm;
    }

    /**
     * Returns the random number generator
     *
     * @return Random number generator as std::string
     */
    std::string getRandomGenerator() {
        return this->randGenerator;
    }
    /**
     * Sets the random number generator
     *
     * @param rand as std::string
     */
    void setRandomGenerator(std::string rand) {
        this->randGenerator = rand;
    }

    /**
     * Returns the seeds for the controls
     *
     * @return Seeds as const std::vector<double>&
     */
    const std::vector<double>& getControlSeeds() {
        return this->seedsControl;
    }
    /**
     * Sets the seeds for the controls
     *
     * @param seeds as const std::vector<double>&
     */
    void setControlSeeds(const std::vector<double>& seeds) {
        this->seedsControl = seeds;
    }

    /**
     * Returns the seeds for exogenous uncertainties
     *
     * @return Seeds as const std::vector<double>&
     */
    const std::vector<double>& getExogenousSeeds() {
        return this->seedsExogenous;
    }
    /**
     * Sets the seeds for exogenous uncertainties
     *
     * @param seeds as const std::vector<double>&
     */
    void setExogenousSeeds(const std::vector<double>& seeds) {
        this->seedsExogenous = seeds;
    }

    /**
     * Returns the seeds for endogenous uncertainties
     *
     * @return Seeds as const std::vector<double>&
     */
    const std::vector<double>& getEndogenousSeeds() {
        return this->seedsEndogenous;
    }
    /**
     * Sets the seeds for endogenous uncertainties
     *
     * @param seeds as const std::vector<double>&
     */
    void setEndogenousSeeds(const std::vector<double>& seeds) {
        this->seedsEndogenous = seeds;
    }

    /**
     * Returns the final valuation
     *
     * @return Value as double
     */
    double getValue() {
        return this->value;
    }
    /**
     * Sets the final valuation
     *
     * @param value as double
     */
    void setValue(double value) {
        this->value = value;
    }

    /**
     * Returns the values of every path generated and optimally controlled
     *
     * @return End values as const Eigen::VectorXd&
     */
    const Eigen::VectorXd& getValues() {
        return this->values;
    }
    /**
     * Sets the values of every path generated and optimally controlled
     *
     * @param values as const Eigen::VectorXd&
     */
    void setValues(const Eigen::VectorXd& values) {
        this->values = values;
    }

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

    /**
     * Runs the optimal control simulation to determine road value
     *
     * This routine computes the Real Options Value of a road over the
     * projected life with the ability to select different options based on the
     * prevailing state. This method uses control randomisation with forward
     * path recomputation (Henry-Labordere and Guyon 2011; Kharroubi, Langrene
     * and Pham 2014) to simulate a number of Monte Carlo paths using a
     * random selection of the available controls at each time step for each
     * path. These are then used in a backward stochastic differential equation
     * to compute conditional expectations for each control at each time step
     * and expected state. The state here consists of the overall population as
     * well as the AAR for each control. This dimension is a measure of the
     * proportion of animals killed by the road from time t to t+1 under each
     * control. It is deterministic because we treate the transition and
     * survival matricies as fixed for a run (they are hidden parameters as
     * opposed to uncertain). Furthermore, using AAR allows us to create an
     * 'adjusted' population for each control, which represents the number of
     * animals left at the end of the run for breeding at the start of the next
     * time step. This means that we can reduce the entire problem down from np
     * states (number of patches) to just one without losing the effect of the
     * animal positions within the grid as would happend if we just considered
     * population alone.
     */
    virtual void simulateROVCR();

    /**
     * Builds the policy map for a given road
     */
    void buildPolicyMap();

    /**
     * Updates the state
     *
     * @param state as StatePtr
     */
    void randomState(StatePtr state);

    /**
     * Simulates all the forward paths
     */
    void simulateForwardPaths();

private:
    StatePtr state;                         /**< State object used in simulation */
    PolicyMapPtr policyMap;                 /**< Generated policy map */
    std::string randGenerator;              /**< Random number generator used */
    std::vector<double> seedsControl;       /**< Seeds for control randomisation */
    std::vector<double> seedsExogenous;     /**< Seeds for exogenous uncertainty */
    std::vector<double> seedsEndogenous;    /**< Seeds for endogenous uncertainty */
    double value;                           /**< Computed value */
    Eigen::VectorXd values;                 /**< Values of all paths (controlled) generated */
    double paths;                           /**< Path values */
    int controls;                           /**< Number of controls */
};

#endif

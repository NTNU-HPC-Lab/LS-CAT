#ifndef ECONOMIC_H
#define ECONOMIC_H

class Commodity;
typedef std::shared_ptr<Commodity> CommodityPtr;

class Economic;
typedef std::shared_ptr<Economic> EconomicPtr;

/**
 * Class for managing economic information
 */
class Economic : public std::enable_shared_from_this<Economic> {

public:
    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////
    /**
     * Constructor I
     *
     * Constructs an empty %Economic object
     */
    Economic();

    /**
     * Constructor II
     *
     * Constructs an %Economic object with assigned values
     */
    Economic(const std::vector<CommodityPtr>& commodities,
            const std::vector<CommodityPtr>& fuels, double rr, double ny,
            double ss);

    /**
     * Destructor
     */
    ~Economic();

    // ACCESSORS //////////////////////////////////////////////////////////////

    /**
     * Returns the commodities used
     *
     * @return Commodities as const std::vector<CommodityPtr>&
     */
    const std::vector<CommodityPtr>& getCommodities() {
        return this->commodities;
    }
    /**
     * Sets the commodities
     *
     * @param comm as const std::vector<CommodityPtr>&
     */
    void setCommodities(const std::vector<CommodityPtr>& comm) {
        this->commodities = comm;
    }

    /**
     * Returns the fuels
     *
     * @return Fuels as const std::vector<CommodityPtr>&
     */
    const std::vector<CommodityPtr>& getFuels() {
        return this->fuels;
    }
    /**
     * Sets the fuels
     *
     * @param fuels as const std::vector<CommodityPtr>&
     */
    void setFuels(const std::vector<CommodityPtr>& fuels) {
        this->fuels = fuels;
    }

    /**
     * Returns the requried rate of return
     *
     * @return Required rate of return p.a. as double
     */
    double getRRR() {
        return this->reqRate;
    }
    /**
     * Sets the required rate of return
     *
     * @param rrr as double
     */
    void setRRR(double rrr) {
        this->reqRate = rrr;
    }

    /**
     * Returns the design horizon in years
     *
     * @return Design horizon as double
     * @note This is actually the number of time steps, not the number of years
     */
    double getYears() {
        return this->nYears;
    }
    /**
     * Sets the design horizon in years
     *
     * @param years as double
     */
    void setYears(double years) {
        this->nYears = years;
    }

    /**
     * Returns the step size used in simulations
     *
     * @return Step size as double
     */
    double getTimeStep() {
        return this->timeStep;
    }
    /**
     * Sets the step size used in simulations
     *
     * @param step as double
     */
    void setTimeStep(double step) {
        this->timeStep = step;
    }

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

private:
    std::vector<CommodityPtr> commodities;  /**< Relevant commodities */
    std::vector<CommodityPtr> fuels;        /**< Relevant fuels */
    double reqRate;                         /**< Required rate of return p.a. */
    double nYears;                          /**< Design horizon (noSteps) */
    double timeStep;                        /**< Simulation step size in years */
};

#endif

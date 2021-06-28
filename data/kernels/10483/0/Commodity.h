#ifndef COMMODITY_H
#define COMMODITY_H

class Optimiser;
typedef std::shared_ptr<Optimiser> OptimiserPtr;

class Uncertainty;
typedef std::shared_ptr<Uncertainty> UncertaintyPtr;

class Commodity;
typedef std::shared_ptr<Commodity> CommodityPtr;

class CommodityCovariance;
typedef std::shared_ptr<CommodityCovariance> CommodityCovariancePtr;

/**
 * Class for managing %Commodity objects (including fuels)
 */
class Commodity : public Uncertainty {

public:
    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    /**
     * Constructor I
     *
     * Constructs a %Commodity object with default values
     */
    Commodity(OptimiserPtr optimiser);

    /**
     * Constructor II
     *
     * Constructs a %Commodity object with assigned values
     */
    Commodity(OptimiserPtr optimiser, std::string nm, double curr, double mp,
            double sd, double rev, double pj, double jp, bool active, double oc,
            double ocsd);

    /**
     * Destructor
     */
    ~Commodity();

    // ACCESSORS //////////////////////////////////////////////////////////////

    /**
     * Returns the commodity's covariance with other commodities
     *
     * @return Covariances as std::vector<CommodityCovariancePtr>&
     */
    const std::vector<CommodityCovariancePtr>& getCovariances() {
        return this->covariances;
    }
    /**
     * Sets the commodity's covariance with other commodities
     *
     * @param covs as std::vector<CommodityCovariancePtr>&
     */
    void setCovariances(const std::vector<CommodityCovariancePtr>& covs) {
        this->covariances = covs;
    }

    /**
     * Returns the ore content.
     *
     * Returns the average proportion of one tonne of dirt of which this
     * commodity comprises.
     *
     * @return Ore content proportion as double
     */
    double getOreContent() {
        return this->oreContent;
    }
    /**
     * Sets the ore content.
     *
     * Sets the average proportion of one tonne of dirt of which this
     * commodity comprises.
     *
     * @param oc as double
     */
    void setOreContent(double oc) {
        this->oreContent = oc;
    }

    /**
     * Returns the ore content standard deviation.
     *
     * Returns the standard deviation of the proportion of one tonne of dirt
     * of which this commodity comprises.
     *
     * @return Ore content proportion standard deviation as double
     */
    double getOreContentSD() {
        return this->oreContentSD;
    }
    /**
     * Sets the ore content standard deviation.
     *
     * Sets the standard deviation of the proportion of one tonne of dirt
     * of which this commodity comprises.
     *
     * @param ocsd as double
     */
    void setOreContentSD(double ocsd) {
        this->oreContentSD = ocsd;
    }

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

private:
    // At this stage ore content per tonne is fixed
    std::vector<CommodityCovariancePtr> covariances;/**< Covariances with other commodities */
    double oreContent;            /**< Mean proportion of raw ore made up by this commodity */
    double oreContentSD;          /**< Standard deviation of amount in ore */
};

#endif

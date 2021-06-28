#ifndef COMMODITYCOVARIANCE_H
#define COMMODITYCOVARIANCE_H

class Commodity;
typedef std::shared_ptr<Commodity> CommodityPtr;

class CommodityCovariance;
typedef std::shared_ptr<CommodityCovariance> CommodityCovariancePtr;

/**
 * Class for managing the covariances of Commodity objects
 */
class CommodityCovariance : public
		std::enable_shared_from_this<CommodityCovariance> {

public:
    // CONSTRUCTORS AND DESTRUCTORS /////////////////////////////////////////////

    /**
     * Constructor
     *
     * Constructs a %CommodityCovariance object
     */
    CommodityCovariance(CommodityPtr com1, CommodityPtr com2, double cov);

    /**
     * Destructor
     */
    ~CommodityCovariance();

    // ACCESSORS ////////////////////////////////////////////////////////////////

    /**
     * Returns the first commodity
     *
     * @return Commodity as CommodityPtr
     */
    CommodityPtr getCommodity1() {
    return this->commodity1.lock();
    }
    /**
     * Sets the first commodity
     *
     * @param comm as CommodityPtr
     */
    void setCommodity1(CommodityPtr comm) {
    this->commodity1.reset();
            this->commodity1 = comm;
    }

    /**
     * Returns the second commodity
     *
     * @return Commodity as CommodityPtr
     */
    CommodityPtr getCommodity2() {
    return this->commodity2.lock();
    }
    /**
     * Sets the second commodity
     *
     * @param comm as CommodityPtr
     */
    void setCommodity2(CommodityPtr comm) {
    this->commodity2.reset();
            this->commodity2 = comm;
    }

    /**
     * Returns the covariance
     *
     * @return Covariance as double
     */
    double getCovariance() {
            return this->covariance;
    }
    /**
     * Sets the covariance
     *
     * @param cov as double
     */
    void setCovariance(double cov) {
            this->covariance = cov;
    }

    // STATIC ROUTINES //////////////////////////////////////////////////////////

    // CALCULATION ROUTINES /////////////////////////////////////////////////////

private:
    std::weak_ptr<Commodity> commodity1;    /**< First Commodity */
    std::weak_ptr<Commodity> commodity2;    /**< Second Commodity */
    double covariance;                      /**< Covariance */
};

#endif

#ifndef TRAFFICPROGRAM_H
#define TRAFFICPROGRAM_H

class Traffic;
typedef std::shared_ptr<Traffic> TrafficPtr;

class TrafficProgram;
typedef std::shared_ptr<TrafficProgram> TrafficProgramPtr;

/**
 * Class for managing %TrafficProgram objects
 */
class TrafficProgram : public Program {

public:
    // CONSTRUCTORS AND DESTRUCTORS ///////////////////////////////////////////

    /**
     * Constructor I
     *
     * Constructs an empty %TrafficProgram object
     */
    TrafficProgram(const Eigen::VectorXd& flowRates, const Eigen::MatrixXd&
        switching);

    /**
     * Constructor II
     *
     * Constructs a %TrafficProgram control object
     */
    TrafficProgram(bool br, TrafficPtr traffic, const Eigen::VectorXd&
        flowRates, const Eigen::MatrixXd &switching);

    /**
     * Destructor
     */
    ~TrafficProgram();

    // ACCESSORS //////////////////////////////////////////////////////////////

    /**
     * Returns whether a bridge option is available
     *
     * @return Bridge as bool
     */
    bool getBridge() {
        return this->bridge;
    }
    /**
     * Sets whether a bridge option is available
     *
     * @param br as bool
     */
    void setBridge(bool br) {
        this->bridge = br;
    }

    /**
     * Returns the Traffic data
     *
     * @return Traffic as TrafficPtr
     */
    TrafficPtr getTraffic() {
        return this->traffic;
    }
    /**
     * Sets the Traffic data
     *
     * @param traffic as TrafficPtr
     */
    void setTraffic(TrafficPtr traffic) {
    this->traffic.reset();
        this->traffic = traffic;
    }

    // STATIC ROUTINES ////////////////////////////////////////////////////////

    // CALCULATION ROUTINES ///////////////////////////////////////////////////

private:
    bool bridge;        /**< Whether there is an option to build an optimal bridge */
    TrafficPtr traffic;	/**< Traffic object used */
};

#endif

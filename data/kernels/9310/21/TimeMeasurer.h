#pragma once

namespace utl
{

class _TimeMeasurer;

class TimeMeasurer
{
public:

    TimeMeasurer();
    ~TimeMeasurer();

    //TimeMeasurer(const TimeMeasurer& inst) = delete;
    //TimeMeasurer operator=(const TimeMeasurer& inst) = delete;

    void start();
    void stop();
    void reset();

    double fetchTimeAvg() const;
    double fetchTimeStd() const;
    double fetchTimeMax() const;
    double fetchTimeMin() const;
    double fetchTimeMedian() const;
    unsigned int fetchCount() const;

    const char* toString() const;

private:
    TimeMeasurer(const TimeMeasurer& inst);
    TimeMeasurer operator=(const TimeMeasurer& inst);

    _TimeMeasurer* m_impl;
};

}

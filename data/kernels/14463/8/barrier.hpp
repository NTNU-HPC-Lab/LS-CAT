/**
 *  Quantum Lattice Boltzmann 
 *  (c) 2015 Fabian Thüring, ETH Zürich
 *
 *  Implementation of a barrier class which is used to synchronize C++11-threads.
 *  The barrier will either spin waiting (SpinBarrier) or use a condition variable
 *  to wait (CondBarrier)
 */
 
#ifndef BARRIER_HPP
#define BARRIER_HPP

#if !defined(_MSC_VER) && __cplusplus <= 199711L
#error "This file requires C++11 support." 
#endif
 
#include <mutex>
#include <thread>
#include <condition_variable>
#include <cassert>

class Barrier
{
public:
    Barrier(int nthread)
    	:	nthread_(nthread), count_(nthread), generation_(0)
    {
    	assert(nthread != 0);
    }
    virtual ~Barrier()
    {}
    
    /**
     *	Wait until all (count) threads have arrived
     */
    virtual void wait() = 0;
    
    /**
     *	Get number of waiting threads
     */
  	int num_waiting() const
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return count_;
    }
    
protected:
    mutable std::mutex mutex_;
    int nthread_;
    int count_;
    int generation_;
};


class SpinBarrier : public Barrier
{
public:
	SpinBarrier(int count)
		:	Barrier(count)
	{}
	
	~SpinBarrier()
	{}

	void wait() override
    {
        std::unique_lock<std::mutex> lock(mutex_);
        int gen = generation_;
        
        if (--count_ == 0) 
        {
            // if done reset to new generation of wait
            count_ = nthread_;
            generation_++;
        }
        else 
        {
			// spin waiting
			lock.unlock();
			while(true) 
			{
				lock.lock();
				if (gen != generation_)
					break;
				lock.unlock();
			}
        }
    }
};


class CondBarrier : public Barrier
{
public:
	CondBarrier(int count)
		:	Barrier(count)
	{}
	
	~CondBarrier()
	{}
	
	void wait() override
	{
		std::unique_lock<std::mutex> lock(mutex_);
		
		if (--count_ == 0)
		{
			count_ = nthread_;
			cond_.notify_all();
		}
		else
			// sleep waiting
			cond_.wait(lock);
	}
	
private:
	std::condition_variable cond_;
};

#endif /* barrier.hpp */

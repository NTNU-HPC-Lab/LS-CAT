#ifndef THREADMANAGER_H
#define THREADMANAGER_H

// For now the code runs on a single machine with shared memory

typedef std::shared_ptr<ctpl::thread_pool> PoolPtr;

class ThreadManager;
typedef std::shared_ptr<ThreadManager> ThreadManagerPtr;

/**
 * @brief The ThreadManager class
 *
 * Class for managing threads for the Road Design software using thread pools
 */
class ThreadManager {

public:

// CONSTRUCTORS ///////////////////////////////////////////////////////////////
    ThreadManager(unsigned long max_threads);

    /**
     * Destructor
     */
    ~ThreadManager();

// ACCESSORS //////////////////////////////////////////////////////////////////
    /**
     * Returns the user-input desired number of threads
     *
     * @return User-requested number of threads as unsigned long
     */
    unsigned long getMaxThreads() {
        return ThreadManager::max_threads;
    }

    /**
     * Returns the actual number of independent threads in the pool
     *
     * @return Number of independent threads in pool as unsigned long
     */
    unsigned long getNoThreads() {
        return ThreadManager::noThreads;
    }

    /**
     * Returns the thread pool
     *
     * @return Thread pool as PoolPtr
     */
    PoolPtr getPool() {
        return this->pool;
    }

// STATIC ROUTINES ////////////////////////////////////////////////////////////

// CALCULATION ROUTINES ///////////////////////////////////////////////////////

    /**
     * Pushes a function onto the thread pool
     *
     * @param F is a function that accepts an integer and then a variable
     * number of other inputs of any typedef
     * @param Rest is the argument list for the input function (excluding the
     * initial int parameter)
     *
     * @return Results as std::vector< std::future< <F> > >
     */
    template<typename F, typename... Rest>
    auto push(F && f, Rest&&... rest) ->std::future<decltype(f(0, rest...))> {
        return this->pool->push(f, rest...);
    }

    /**
     * Empties the thread pool queue
     */
    void clearQueue() {
        this->pool->clear_queue();
    }

    /**
     * Locks the access to parallel code to prevent concurrent access
     */
    void lock() {
        this->mtx.lock();
    }
    /**
     * Unlocks to allow concurrent access
     */
    void unlock() {
        this->mtx.unlock();
    }

///////////////////////////////////////////////////////////////////////////////
private:
    unsigned long max_threads;      /**< User-defined number of threads */
    unsigned long noThreads;        /**< Actual number of threads */
    std::mutex mtx;                 /**< Mutex for writing out information */
    PoolPtr pool;                   /**< Pool for managing threads */
};
#endif

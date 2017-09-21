#ifndef _H_LOFTE_MAIN_POOL
#define _H_LOFTE_MAIN_POOL

#include <memory>
#include <thread>
#include <vector>

#include "config.hpp"
#include "gpu_pool.cuh"

/*! \class MainPool
    \brief Main pool class, containter for GpuPool(s).

*/

class MainPool
{
    private:

        int nogpus;

        std::vector<std::unique_ptr<GpuPool>> gpuvector;
        std::vector<std::thread> threadvector;
    protected:

    public:
        MainPool(void) = delete;
        MainPool(InConfig config);
        MainPool(const MainPool &inpool) = delete;
        MainPool& operator=(const MainPool &inpool) = delete;
        MainPool(MainPool &&inpool) = delete;
        MainPool& operator=(MainPool &&inpool) = delete;
        ~MainPool(void);
};

#endif

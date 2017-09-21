#include <memory>
#include <thread>
#include <utility>
#include <vector>

using std::move;
using std::thread;
using std::unique_ptr;
using std::vector;

MainPool::MainPool(InConfig config) : nogpus(config.nogpus) {

    for (int igpu = 0; igpu < nogpus; ++igpu) {
        gpuvector.push_back(unique_ptr<GPUpool>(new GPUpool(igpu, config)));
    }

    for (int igpu = 0; igpu < nogpus; ++igpu) {
        threadvector.push_back(thread(&GPUpool::Initialise, std::move(gpuvector[igpu])));
    }

}

MainPool::~MainPool(void) {
    for (int igpu = 0; igpu < ngpus; ++igpu) {
        threadvector[igpu].join();
    }
}

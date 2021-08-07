#include <iostream>
#include <fstream>
#include <chrono>
#include "Timer.hpp"

#include <graphblas/graphblas.hpp>
#include <algorithms/page_rank.hpp>

#include <metall/metall.hpp>
#include <metall/utility/fallback_allocator_adaptor.hpp>

//****************************************************************************
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "ERROR: too few arguments." << std::endl;
        exit(1);
    }
    // Read the numnodes
    
    grb::IndexType NUM_NODES = atoi(argv[1]);

    Timer<std::chrono::steady_clock, std::chrono::milliseconds> my_timer;
    using T = int32_t;

    using allocator_t = metall::utility::fallback_allocator_adaptor<metall::manager::allocator_type<char>>;
    using Metall_MatType = grb::Matrix<T, allocator_t>;


    //================= Page Rank in Metall Scope =========================

    {
        my_timer.start();
        metall::manager manager(metall::open_read_only, "/mnt/ssd/datastore");
        Metall_MatType *A = manager.find<Metall_MatType>("gbtl_vov_matrix").first;
        my_timer.stop();
        std::cout << "Page rank re-attach time: \t\t" << my_timer.elapsed() 
                  << " milli seconds." << std::endl;

        my_timer.start();
        grb::Vector<double> page_rank(NUM_NODES);
        algorithms::page_rank(*A, page_rank);
        my_timer.stop();
        std::cout << "Page rank Algorithm time: \t\t" << my_timer.elapsed() << " milli seconds." << std::endl;

    }
    return 0;
}


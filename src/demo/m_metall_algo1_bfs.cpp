#include <iostream>
#include <fstream>
#include <chrono>
#include "Timer.hpp"

#include <graphblas/graphblas.hpp>
#include <algorithms/triangle_count.hpp>
#include <algorithms/sssp.hpp>
#include <algorithms/bfs.hpp>
#include <algorithms/cluster_louvain.hpp>
#include <algorithms/k_truss.hpp>

#include <metall/metall.hpp>
#include <metall_utility/fallback_allocator_adaptor.hpp>

//****************************************************************************
int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cerr << "ERROR: too few arguments." << std::endl;
        exit(1);
    }
    using T = int32_t;
    // Read the numnodes
    
    grb::IndexType NUM_NODES = atoi(argv[1]);
    T iA_front = atoi(argv[2]);
    T jA_front = atoi(argv[3]);
    

    Timer<std::chrono::steady_clock, std::chrono::milliseconds> my_timer;

    using allocator_t = metall::utility::fallback_allocator_adaptor<metall::manager::allocator_type<char>>;
    using Metall_MatType = grb::Matrix<T, allocator_t>;



    //================= single BFS in Metall Scope ================================
    {
        my_timer.start();
        metall::manager manager(metall::open_read_only, "/mnt/ssd/datastore");
        Metall_MatType *A = manager.find<Metall_MatType>("gbtl_vov_matrix").first;
        my_timer.stop();
        std::cout << "BFS re-attach time: \t\t" << my_timer.elapsed() 
                  << " milli seconds." << std::endl;

        my_timer.start();
        grb::Vector<T> parent_list(NUM_NODES);
        grb::Vector<T> root(NUM_NODES);
        root.setElement(iA_front, jA_front);
        algorithms::bfs(*A, root, parent_list);
        my_timer.stop();
        std::cout << "BFS Algorithm time: \t\t" << my_timer.elapsed() 
                  << " milli seconds." << std::endl;

        //grb::print_vector(std::cout, parent_list, "Parent list for root at vertex 3");
    }
    
    return 0;
}


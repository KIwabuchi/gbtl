/*
 * GraphBLAS Template Library (GBTL), Version 3.0
 *
 * Copyright 2020 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors.
 *
 * THIS MATERIAL WAS PREPARED AS AN ACCOUNT OF WORK SPONSORED BY AN AGENCY OF
 * THE UNITED STATES GOVERNMENT.  NEITHER THE UNITED STATES GOVERNMENT NOR THE
 * UNITED STATES DEPARTMENT OF ENERGY, NOR THE UNITED STATES DEPARTMENT OF
 * DEFENSE, NOR CARNEGIE MELLON UNIVERSITY, NOR BATTELLE, NOR ANY OF THEIR
 * EMPLOYEES, NOR ANY JURISDICTION OR ORGANIZATION THAT HAS COOPERATED IN THE
 * DEVELOPMENT OF THESE MATERIALS, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS,
 * OR USEFULNESS OR ANY INFORMATION, APPARATUS, PRODUCT, SOFTWARE, OR PROCESS
 * DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED
 * RIGHTS.
 *
 * Released under a BSD-style license, please see LICENSE file or contact
 * permission@sei.cmu.edu for full terms.
 *
 * [DISTRIBUTION STATEMENT A] This material has been approved for public release
 * and unlimited distribution.  Please see Copyright notice for non-US
 * Government use and distribution.
 *
 * DM20-0442
 */

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
    if (argc < 2)
    {
        std::cerr << "ERROR: too few arguments." << std::endl;
        exit(1);
    }
    // Read the edgelist and create the tuple arrays
    std::string pathname(argv[1]);

    Timer<std::chrono::steady_clock, std::chrono::milliseconds> my_timer;

    grb::IndexArrayType iL, iU, iA;
    grb::IndexArrayType jL, jU, jA;
    uint64_t num_rows = 0;
    uint64_t max_id = 0;
    uint64_t src, dst;

    my_timer.start();
    {
        std::ifstream infile(pathname);
        while (infile)
        {
            infile >> src >> dst;
            //std::cout << "Read: " << src << ", " << dst << std::endl;
            if (src > max_id) max_id = src;
            if (dst > max_id) max_id = dst;

            if (src < dst)
            {
                iA.push_back(src);
                jA.push_back(dst);

                iU.push_back(src);
                jU.push_back(dst);
            }
            else if (dst < src)
            {
                iA.push_back(src);
                jA.push_back(dst);

                iL.push_back(src);
                jL.push_back(dst);
            }
            // else ignore self loops

            ++num_rows;
        }
    }

    std::cout << "Read " << num_rows << " rows." << std::endl;
    std::cout << "#Nodes = " << (max_id + 1) << std::endl;

    // sort the
    using DegIdx = std::tuple<grb::IndexType,grb::IndexType>;
    std::vector<DegIdx> degrees(max_id + 1);
    for (grb::IndexType idx = 0; idx <= max_id; ++idx)
    {
        degrees[idx] = {0UL, idx};
    }

    {
        std::ifstream infile(pathname);
        while (infile)
        {
            infile >> src >> dst;
            if (src != dst)
            {
                std::get<0>(degrees[src]) += 1;
            }
        }
    }

    std::sort(degrees.begin(), degrees.end(),
              [](DegIdx a, DegIdx b) { return std::get<0>(b) < std::get<0>(a); });

    //relabel
   
    for (auto &idx : iL) { idx = std::get<1>(degrees[idx]); }
    for (auto &idx : jL) { idx = std::get<1>(degrees[idx]); }


    using T = int32_t;
    using allocator_t = metall::utility::fallback_allocator_adaptor<metall::manager::allocator_type<char>>;
    using Metall_MatType = grb::Matrix<T, allocator_t>;

    grb::IndexType NUM_NODES(max_id + 1);
    std::vector<T> v(iA.size(), 1);

    //================= Graph Construction in Metall Scope ========================

    {
        metall::manager manager(metall::create_only, "/dev/shm/datastore");
        Metall_MatType *A = manager.construct<Metall_MatType>("gbtl_vov_matrix")
                        ( NUM_NODES, NUM_NODES, manager.get_allocator());
        A->build(iA.begin(), jA.begin(), v.begin(), iA.size());
    }
    my_timer.stop();
    std::cout << "Graph Construction time: \t" << my_timer.elapsed() 
                << " milli seconds." << std::endl;  
    

    //================= Triangle Counting in Metall Scope =========================

    my_timer.start();
    {
        metall::manager manager(metall::open_only, "/dev/shm/datastore");
        Metall_MatType *A = manager.find<Metall_MatType>("gbtl_vov_matrix").first;
        T count(0);
        count = algorithms::triangle_count_masked_noT(*A);
    }
    my_timer.stop();
    std::cout << "TC Algorithm time: \t\t" << my_timer.elapsed() << " milli seconds." << std::endl;

    //================= single BFS in Metall Scope ================================
    my_timer.start();
    {
        metall::manager manager(metall::open_only, "/dev/shm/datastore");
        Metall_MatType *A = manager.find<Metall_MatType>("gbtl_vov_matrix").first;
        grb::Vector<T> parent_list(NUM_NODES);
        grb::Vector<T> root(NUM_NODES);
        root.setElement(iA.front(), jA.front());
        algorithms::bfs(*A, root, parent_list);
        //grb::print_vector(std::cout, parent_list, "Parent list for root at vertex 3");
    }
    my_timer.stop();
    std::cout << "BFS Algorithm time: \t\t" << my_timer.elapsed() 
                << " milli seconds." << std::endl;



    return 0;


}

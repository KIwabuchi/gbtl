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
#include <cstdio>
#include <chrono>
#include "Timer.hpp"
#include <graphblas/graphblas.hpp>
#include <algorithms/k_truss.hpp>

using namespace grb;

//****************************************************************************
namespace
{
    IndexArrayType i = {
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33};

    IndexArrayType j = {
        1,2,3,4,5,6,7,8,10,11,12,13,17,19,21,31,     
        8,9,13,14,15,18,19,20,22,23,26,27,28,29,30,31,32};
}

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


    Timer<std::chrono::steady_clock, std::chrono::microseconds> my_timer;
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
    my_timer.stop();
    std::cout << "Elapsed read time: " << my_timer.elapsed() << " usec." << std::endl;

    std::cout << "Read " << num_rows << " rows." << std::endl;
    std::cout << "#Nodes = " << (max_id + 1) << std::endl;

    my_timer.start();

    using T = int;

    // create an incidence matrix from the data
    IndexType num_edges = 0;
    IndexType num_nodes = 0;
    IndexArrayType edge_array, node_array;
    // count edges in upper triangle of A
    for (IndexType ix = 0; ix < iA.size(); ++ix)
    {
        if (i[ix] < j[ix])
        {
            edge_array.push_back(num_edges);
            node_array.push_back(iA[ix]);
            edge_array.push_back(num_edges);
            node_array.push_back(jA[ix]);
            ++num_edges;

            num_nodes = std::max(num_nodes, iA[ix]);
            num_nodes = std::max(num_nodes, jA[ix]);
        }
    }
    ++num_nodes;
    std::vector<T> v(edge_array.size(), 1);

    Matrix<T> E(num_edges, num_nodes);
    E.build(edge_array.begin(), node_array.begin(), v.begin(), v.size());
    
    my_timer.stop();
    std::cout << "Graph Construction time: " << my_timer.elapsed() << " usec." << std::endl;
 

    std::cout << "Running k-truss algorithm..." << std::endl;

    my_timer.start();

    auto Eout3 = algorithms::k_truss(E, 3); 
    my_timer.stop();
    std::cout << "algorithms Edges in 3-trusses: " << my_timer.elapsed() << " usec." << std::endl;

       
    return 0;
}

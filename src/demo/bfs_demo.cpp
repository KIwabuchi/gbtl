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

#include <graphblas/graphblas.hpp>
#include <algorithms/bfs.hpp>

#include <metall/metall.hpp>
#include <metall_utility/fallback_allocator_adaptor.hpp>

//****************************************************************************
int main()
{
    // FA Tech Note graph
    //
    //    {-, -, -, 1, -, -, -, -, -},
    //    {-, -, -, 1, -, -, 1, -, -},
    //    {-, -, -, -, 1, 1, 1, -, 1},
    //    {1, 1, -, -, 1, -, 1, -, -},
    //    {-, -, 1, 1, -, -, -, -, 1},
    //    {-, -, 1, -, -, -, -, -, -},
    //    {-, 1, 1, 1, -, -, -, -, -},
    //    {-, -, -, -, -, -, -, -, -},
    //    {-, -, 1, -, 1, -, -, -, -};

    using T = grb::IndexType;

    grb::IndexType const NUM_NODES = 9;
    grb::IndexArrayType i = {0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                             4, 4, 4, 5, 6, 6, 6, 8, 8};
    grb::IndexArrayType j = {3, 3, 6, 4, 5, 6, 8, 0, 1, 4, 6,
                             2, 3, 8, 2, 1, 2, 3, 2, 4};
    std::vector<T> v(i.size(), 1);

    /// @todo change scalar type to unsigned int or grb::IndexType

    using allocator_t = metall::utility::fallback_allocator_adaptor<metall::manager::allocator_type<char>>;
    using Metall_GBMatrix = grb::Matrix<T, allocator_t>;
    {
        std::cout<<"Entering Metall Scope"<<std::endl;
        metall::manager manager(metall::create_only, "/tmp/kaushik/datastore");
        Metall_GBMatrix *G_tn = manager.construct<Metall_GBMatrix>("gbtl_vov_matrix")
                        ( NUM_NODES, NUM_NODES, manager.get_allocator());
        G_tn->build(i.begin(), j.begin(), v.begin(), i.size());
        grb::print_matrix(std::cout, *G_tn, "Graph adjacency matrix:");
        std::cout<<"Exiting Metall Scope"<<std::endl;
    }

    {
        std::cout<<"Entering Metall Scope"<<std::endl;
        metall::manager manager(metall::open_only, "/tmp/kaushik/datastore");
        Metall_GBMatrix *G_tn = manager.find<Metall_GBMatrix>("gbtl_vov_matrix").first;
        grb::Vector<T> parent_list(NUM_NODES);
        grb::Vector<T> root(NUM_NODES);
        root.setElement(3, 1);
        // Perform a single BFS
        algorithms::bfs(*G_tn, root, parent_list);
        grb::print_vector(std::cout, parent_list, "Parent list for root at vertex 3");
        std::cout<<"Exiting Metall Scope"<<std::endl;
    }

    {
        std::cout<<"Entering Metall Scope"<<std::endl;
        metall::manager manager(metall::open_only, "/tmp/kaushik/datastore");
        Metall_GBMatrix *G_tn = manager.find<Metall_GBMatrix>("gbtl_vov_matrix").first;
        // Perform BFS from all roots simultaneously (should the value be 0?)
        //auto roots = grb::identity<GBMatrix>(NUM_NODES, INF, 0);
        Metall_GBMatrix roots(NUM_NODES, NUM_NODES);
        Metall_GBMatrix G_tn_res(NUM_NODES, NUM_NODES);
        grb::IndexArrayType ii, jj, vv;
        for (grb::IndexType ix = 0; ix < NUM_NODES; ++ix)
        {
            ii.push_back(ix);
            jj.push_back(ix);
            vv.push_back(1);
        }
        roots.build(ii, jj, vv);
        algorithms::bfs_batch(*G_tn, roots, G_tn_res);
        grb::print_matrix(std::cout, G_tn_res, "Parents for each root (by rows):");
        std::cout<<"Exiting Metall Scope"<<std::endl;
    }

    return 0;
}

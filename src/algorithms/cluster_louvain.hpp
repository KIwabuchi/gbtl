/*
 * GraphBLAS Template Library, Version 2.0
 *
 * Copyright 2019 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors. All Rights Reserved.
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
 * RIGHTS..
 *
 * Released under a BSD (SEI)-style license, please see license.txt or contact
 * permission@sei.cmu.edu for full terms.
 *
 * This release is an update of:
 *
 * 1. GraphBLAS Template Library (GBTL)
 * (https://github.com/cmu-sei/gbtl/blob/1.0.0/LICENSE) Copyright 2015 Carnegie
 * Mellon University and The Trustees of Indiana. DM17-0037, DM-0002659
 *
 * DM18-0559
 */

#ifndef ALGORITHMS_CLUSTER_LOUVAINE_HPP
#define ALGORITHMS_CLUSTER_LOUVAINE_HPP

#include <vector>
#include <random>
//#include <math.h>
#include <graphblas/graphblas.hpp>
#include <graphblas/matrix_utils.hpp>

//****************************************************************************
namespace
{
    //************************************************************************
    // Return a random value that is scaled by the value passed in
    template <typename T=float>
    class SetRandom
    {
    public:
        typedef T result_type;
        SetRandom(double seed = 0.) { m_generator.seed(seed); }

        inline result_type operator()(T val)
        {
            return static_cast<T>(val*m_distribution(m_generator) + 0.0001);
        }

    private:
        std::default_random_engine             m_generator;
        std::uniform_real_distribution<double> m_distribution;
    };
}

//****************************************************************************
namespace algorithms
{
    //************************************************************************
    /**
     * @brief Assign a zero-based cluster ID to each vertex based on the
     *        cluster matrix that was output by the louvain clustering algorithm.
     *
     * @param[in]  cluster_matrix  Matrix output from louvain clustering
     *                             algorithms.  Each row corresponds to a
     *                             vertex and each column corresponds to a cluster.
     *
     * @return An array with each vertex's cluster assignment (the row index for
     *         the maximum value).  MAX_UINT is returned if no cluster was
     *         assigned.
     */
    template <typename MatrixT>
    GraphBLAS::Vector<GraphBLAS::IndexType> get_louvain_cluster_assignments(
        MatrixT const &cluster_matrix)
    {
        GraphBLAS::IndexType num_clusters(cluster_matrix.nrows());
        GraphBLAS::IndexType num_nodes(cluster_matrix.ncols());

        GraphBLAS::Vector<GraphBLAS::IndexType> clusters(num_nodes);
        GraphBLAS::Vector<GraphBLAS::IndexType> index_of_vec(num_clusters);
        std::vector<GraphBLAS::IndexType> indices;
        for (GraphBLAS::IndexType ix=0; ix<num_clusters; ++ix)
        {
            indices.push_back(ix);
        }
        index_of_vec.build(indices, indices);

        // return a GraphBLAS::Vector with cluster assignments
        GraphBLAS::mxv(clusters,
                       GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                       GraphBLAS::MaxSelect2ndSemiring<GraphBLAS::IndexType>(),
                       cluster_matrix, index_of_vec);

        return clusters;
    }

    //************************************************************************
    /**
     * @brief Compute the clusters in the given graph using louvain clustering.
     *
     * @param[in]     graph  The graph to compute the clusters on.  Can be a
     *                       weighted graph.
     * @param[in]  max_iters The maximum number of iterations to run if
     *                       convergence doesn't occur first (oscillation uncommon)
     *
     * @return A matrix whose columns correspond to the vertices, and vertices
     *         with the same (max) value in a given row belong to the
     *         same cluster.
     */
    template<typename MatrixT, typename RealT=double>
    GraphBLAS::Matrix<bool> louvain_cluster(
        MatrixT const            &graph,
        unsigned int  max_iters = std::numeric_limits<unsigned int>::max())
    {
        using T = typename MatrixT::ScalarType;
        using RealMatrixT = GraphBLAS::Matrix<RealT>;

        GraphBLAS::IndexType rows(graph.nrows());
        GraphBLAS::IndexType cols(graph.ncols());
        if (rows != cols)
        {
            throw GraphBLAS::DimensionException();
        }

        GraphBLAS::print_matrix(std::cout, graph, "*** graph ***");

        // precompute A + A'
        GraphBLAS::Matrix<T> ApAT(graph);
        GraphBLAS::transpose(ApAT, GraphBLAS::NoMask(), GraphBLAS::Plus<T>(),
                             graph, GraphBLAS::MERGE);

        GraphBLAS::print_matrix(std::cout, ApAT, "*** A+A' ***");

        // k = A * vec(1)  (arithmetric row reduce of adj. matrix)
        GraphBLAS::Vector<T> k(rows);
        GraphBLAS::reduce(k, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                          GraphBLAS::Plus<T>(), graph, GraphBLAS::REPLACE);

        // m = 0.5*k'*vec(1) (reduce k to scalar)
        T m(0);
        GraphBLAS::reduce(m, GraphBLAS::NoAccumulate(),
                          GraphBLAS::PlusMonoid<T>(), k);
        m = m/2.0;

        // Initialize S to identity?
        auto S(GraphBLAS::scaled_identity<GraphBLAS::Matrix<bool>>(rows));
        // HOT FIX
        //GraphBLAS::Matrix<bool> S_row(rows, rows);
        GraphBLAS::Vector<bool> S_row(rows);
        //END HOT FIX

        // create a dense mask vector of all ones (a lame way)
        GraphBLAS::Vector<bool> mask(rows);
        GraphBLAS::reduce(mask, GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                          GraphBLAS::LogicalOr<bool>(), S, GraphBLAS::REPLACE);

        SetRandom<RealT> set_random(11);
        bool vertices_changed(true);

        // repeat while modularity is increasing
        do
        {
            vertices_changed = false;

            for (GraphBLAS::IndexType i = 0; i < rows; ++i)
            {
                // only perform the iteration if the ith vertex is not isolated
                if (k.hasElement(i))
                {
                    std::cout << "===Start of vertex " << i << std::endl;

                    // Replace the following with apply with binary op and scalar
                    GraphBLAS::BinaryOp_Bind2nd<RealT, GraphBLAS::Times<T>>
                        neg_scale_k(static_cast<RealT>(-k.extractElement(i)/m));

                    // v' = e_i' * (A + A') == extract row i of (A + A')
                    GraphBLAS::Vector<RealT> v(rows);
                    GraphBLAS::extract(v, GraphBLAS::NoMask(),
                                       GraphBLAS::NoAccumulate(),
                                       ApAT,
                                       GraphBLAS::AllIndices(), i,
                                       GraphBLAS::REPLACE);

                    // v' += (-k_i/m)*k'
                    GraphBLAS::apply(v, GraphBLAS::NoMask(),
                                     GraphBLAS::Plus<RealT>(),
                                     neg_scale_k, k, GraphBLAS::MERGE);


                    // S := (I - e_i*e_i')S means clear i-th row of S (using a mask)
                    GraphBLAS::Matrix<bool> Mask(rows, rows);
                    GraphBLAS::assign(Mask, GraphBLAS::NoMask(),
                                      GraphBLAS::NoAccumulate(),
                                      mask, i, GraphBLAS::AllIndices(),
                                      GraphBLAS::REPLACE);
                    // HOT FIX
                    //GraphBLAS::apply(S_row, Mask, GraphBLAS::NoAccumulate(),
                    //                 GraphBLAS::Identity<bool>(),
                    //                 S, GraphBLAS::REPLACE);
                    GraphBLAS::extract(S_row, GraphBLAS::NoMask(),
                                       GraphBLAS::NoAccumulate(),
                                       GraphBLAS::transpose(S),
                                       GraphBLAS::AllIndices(), i,
                                       GraphBLAS::REPLACE);
                    //END HOT FIX

                    GraphBLAS::apply(S, GraphBLAS::complement(Mask),
                                     GraphBLAS::NoAccumulate(),
                                     GraphBLAS::Identity<bool>(),
                                     S, GraphBLAS::REPLACE);

                    // q' = v' * S
                    GraphBLAS::Vector<RealT> q(rows);
                    GraphBLAS::vxm(q, GraphBLAS::NoMask(),
                                   GraphBLAS::NoAccumulate(),
                                   GraphBLAS::ArithmeticSemiring<RealT>(),
                                   v, S, GraphBLAS::REPLACE);
                    GraphBLAS::print_vector(std::cout, q, "modularity change");


                    //HOT FIX
                    //extract row i of S to find which original community
                    //GraphBLAS::Vector<RealT> qMask(rows);
                    //GraphBLAS::Vector<bool> qBool(rows);
                    //GraphBLAS::extract(qMask, GraphBLAS::NoMask(),
                    //                   GraphBLAS::NoAccumulate(),
                    //                   GraphBLAS::transpose(S_row),
                    //                   GraphBLAS::AllIndices(), i,
                    //                   GraphBLAS::REPLACE);
                    //GraphBLAS::apply(qBool, qMask, GraphBLAS::NoAccumulate(),
                    //                 GraphBLAS::Identity<bool>(),
                    //                 q, GraphBLAS::REPLACE);
                    //END HOT FIX


                    // kappa = max(q)
                    RealT kappa(0);
                    GraphBLAS::reduce(kappa, GraphBLAS::NoAccumulate(),
                                      GraphBLAS::MaxMonoid<RealT>(), q);

                    // t = (q == kappa)
                    // Replace the following with apply with binary op and scalar
                    GraphBLAS::BinaryOp_Bind2nd<RealT, GraphBLAS::Equal<RealT>>
                        equal_kappa(kappa);

                    GraphBLAS::Vector<bool> t(rows);
                    GraphBLAS::apply(t, GraphBLAS::NoMask(),
                                     GraphBLAS::NoAccumulate(),
                                     equal_kappa, q, GraphBLAS::REPLACE);
                    GraphBLAS::apply(t, t, GraphBLAS::NoAccumulate(),
                                     GraphBLAS::Identity<T>(), t, GraphBLAS::REPLACE);

                    // break ties if necessary
                    while (t.nvals() != 1)
                    {
                        GraphBLAS::print_vector(std::cout, t, "breaking ties");

                        // Assign a random number to each possible cluster
                        GraphBLAS::Vector<RealT> p(rows);
                        GraphBLAS::apply(p, GraphBLAS::NoMask(),
                                         GraphBLAS::NoAccumulate(),
                                         set_random, t, GraphBLAS::REPLACE);
                        // max_p = max(p)
                        RealT max_p(0);
                        GraphBLAS::reduce(max_p, GraphBLAS::NoAccumulate(),
                                          GraphBLAS::MaxMonoid<RealT>(), p);

                        // t = (q == kappa)
                        // Replace the following with apply with binary op and scalar
                        GraphBLAS::BinaryOp_Bind2nd<RealT, GraphBLAS::Equal<RealT>>
                            equal_max_p(max_p);

                        GraphBLAS::apply(t, GraphBLAS::NoMask(),
                                         GraphBLAS::NoAccumulate(),
                                         equal_max_p, p, GraphBLAS::REPLACE);
                        GraphBLAS::apply(t, t, GraphBLAS::NoAccumulate(),
                                         GraphBLAS::Identity<T>(), t,
                                         GraphBLAS::REPLACE);
                    }

                    // Replace row i of S with t
                    GraphBLAS::assign(S, GraphBLAS::NoMask(),
                                      GraphBLAS::NoAccumulate(),
                                      t, i, GraphBLAS::AllIndices(),
                                      GraphBLAS::MERGE);

                    //HOT FIX - compare new community w/ previous community to see if it has changed
                    if (t != S_row) // vertex changed communities
                    {
                        vertices_changed = true;
                        GraphBLAS::print_matrix(std::cout, S, "*** new S ***");
                    }
                    else
                    {
                        std::cout<< "No change. "<< std::endl;
                    }
                    //END HOT FIX
                }
                std::cout << "===End   of vertex " << i << std::endl;
            }

        } while (vertices_changed);

        return S;
    }

} // algorithms

#endif // CLUSTER_HPP

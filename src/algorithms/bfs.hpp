/*
 * GraphBLAS Template Library, Version 2.1
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
 * (https://github.com/cmu-sei/gbtl/blob/master/LICENSE) Copyright 2015 Carnegie
 * Mellon University and The Trustees of Indiana. DM17-0037, DM-0002659
 *
 * DM18-0559
 */

/**
 * @file bfs.hpp
 *
 * @brief Standard breadth first search implementation.
 *
 * <p>Breadth first search implementation, verification function, and human
 * readable error code conversion.</p>
 */

#ifndef ALGORITHMS_BFS_HPP
#define ALGORITHMS_BFS_HPP

#include <limits>
#include <tuple>

#include <graphblas/graphblas.hpp>

//****************************************************************************
namespace
{
    //************************************************************************
    // convert each stored entry in a matrix to its 1-based column index
    template <typename MatrixT>
    void col_index_of_1based(MatrixT &mat)
    {
        using T = typename MatrixT::ScalarType;

        GraphBLAS::IndexArrayType i, j, v;
        for (GraphBLAS::IndexType ix = 0; ix < mat.ncols(); ++ix)
        {
            i.push_back(ix);
            j.push_back(ix);
            v.push_back(ix + 1);
        }

        MatrixT identity_ramp(mat.ncols(), mat.ncols());

        identity_ramp.build(i,j,v);

        GraphBLAS::mxm(mat,
                       GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                       GraphBLAS::MinSelect2ndSemiring<T>(),
                       mat, identity_ramp, GraphBLAS::REPLACE);
    }

    //************************************************************************
    //convert each stored entry of a vector to its 1-based index
    template <typename VectorT>
    void index_of_0based(VectorT &vec)
    {
        using T = typename VectorT::ScalarType;

        GraphBLAS::IndexArrayType i, j, v;
        for (GraphBLAS::IndexType ix = 0; ix < vec.size(); ++ix)
        {
            i.push_back(ix);
            j.push_back(ix);
            v.push_back(ix);
        }

        GraphBLAS::Matrix<T> identity_ramp(vec.size(), vec.size());

        identity_ramp.build(i,j,v);

        GraphBLAS::vxm(vec,
                       GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                       GraphBLAS::MinSelect2ndSemiring<T>(),
                       vec, identity_ramp, GraphBLAS::REPLACE);
    }

    //************************************************************************
    //convert each stored entry of a vector to its 1-based index
    template <typename VectorT>
    void index_of_1based(VectorT &vec)
    {
        using T = typename VectorT::ScalarType;

        GraphBLAS::IndexArrayType i, j, v;
        for (GraphBLAS::IndexType ix = 0; ix < vec.size(); ++ix)
        {
            i.push_back(ix);
            j.push_back(ix);
            v.push_back(ix + 1);
        }

        GraphBLAS::Matrix<T> identity_ramp(vec.size(), vec.size());

        identity_ramp.build(i,j,v);

        GraphBLAS::vxm(vec,
                       GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                       GraphBLAS::MinSelect2ndSemiring<T>(),
                       vec, identity_ramp, GraphBLAS::REPLACE);
    }

}

//****************************************************************************
namespace algorithms
{
    //************************************************************************
    /**
     * @brief Perform a single "parent" breadth first search (BFS) traversal
     *        on the given graph.
     *
     * @param[in]  graph        N x N adjacency matrix of the graph on which to
     *                          perform a BFS. (NOT the transpose).  The value
     *                          1 should indicate an edge.
     * @param[in]  source       Index of the root vertex to use in the
     *                          calculation.
     * @param[out] parent_list  The list of parents for each traversal (row)
     *                          specified in the roots array.
     */
    template <typename MatrixT,
              typename ParentListVectorT>
    void bfs(MatrixT const          &graph,
             GraphBLAS::IndexType    source,
             ParentListVectorT      &parent_list)
    {
        using T = typename MatrixT::ScalarType;
        GraphBLAS::IndexType const N(graph.nrows());

        // assert parent_list is N-vector
        // assert source is in proper range
        // assert parent_list ScalarType is GraphBLAS::IndexType

        // create index ramp for index_of() functionality
        GraphBLAS::Vector<GraphBLAS::IndexType> index_ramp(N);
        {
            std::vector<GraphBLAS::IndexType> idx(N);
            std::vector<GraphBLAS::IndexType> idx1(N);
            for (GraphBLAS::IndexType i = 0; i < N; ++i)
            {
                idx[i] = i;
                idx1[i]= i + 1;  // off-by-one until structure only mask
            }

            index_ramp.build(idx.begin(), idx1.begin(), N);
        }

        // initialize wavefront to source node.
        GraphBLAS::Vector<GraphBLAS::IndexType> wavefront(N);
        wavefront.setElement(source, 1UL);

        // set root parent to self; off-by-one until structure only mask
        parent_list.clear();
        parent_list.setElement(source, source + 1);

        while (wavefront.nvals() > 0)
        {
            // convert all stored values to their 1-based column index
            GraphBLAS::eWiseMult(wavefront,
                                 GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                 GraphBLAS::First<GraphBLAS::IndexType>(),
                                 index_ramp, wavefront);

            // Select1st because we are left multiplying wavefront rows
            // Masking out the parent list ensures wavefront values do not
            // overlap values already stored in the parent list
            GraphBLAS::vxm(wavefront,
                           GraphBLAS::complement(parent_list),
                           GraphBLAS::NoAccumulate(),
                           GraphBLAS::MinSelect1stSemiring<GraphBLAS::IndexType>(),
                           wavefront, graph, GraphBLAS::REPLACE);

            // We don't need to mask here since we did it in mxm.
            // Merges new parents in current wavefront with existing parents
            // parent_list<!parent_list,merge> += wavefront
            GraphBLAS::apply(parent_list,
                             GraphBLAS::NoMask(),
                             GraphBLAS::Plus<GraphBLAS::IndexType>(),
                             GraphBLAS::Identity<GraphBLAS::IndexType>(),
                             wavefront,
                             GraphBLAS::MERGE);
        }

        // REMOVE THE FOLLOWING SUBTRACTION WHEN STRUCTURE ONLY MASKS IMPL'ED
        // Restore zero-based indices by subtracting 1 from all stored values
        GraphBLAS::BinaryOp_Bind2nd<unsigned int,
                                    GraphBLAS::Minus<unsigned int>>
            subtract_1(1);

        GraphBLAS::apply(parent_list,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         subtract_1,
                         parent_list,
                         GraphBLAS::REPLACE);
    }

    //************************************************************************
    /**
     * @brief Perform a single "parent" breadth first search (BFS) traversal
     *        on the given graph.
     *
     * @param[in]  graph        N x N adjacency matrix of the graph on which to
     *                          perform a BFS. (NOT the transpose).  The value
     *                          1 should indicate an edge.
     * @param[in]  wavefront    N-vector, initial wavefront/root to use in the
     *                          calculation. It should have a
     *                          single value set to '1' corresponding to the
     *                          root.
     * @param[out] parent_list  The list of parents for each traversal (row)
     *                          specified in the roots array.
     */
    template <typename MatrixT,
              typename WavefrontVectorT,
              typename ParentListVectorT>
    void bfs(MatrixT const          &graph,
             WavefrontVectorT        wavefront,   // copy is intentional
             ParentListVectorT      &parent_list)
    {
        using T = typename MatrixT::ScalarType;
        GraphBLAS::IndexType const N(graph.nrows());

        // assert parent_list is N-vector
        // assert wavefront is N-vector
        // assert parent_list ScalarType is GraphBLAS::IndexType

        // create index ramp for index_of() functionality
        GraphBLAS::Vector<GraphBLAS::IndexType> index_ramp(N);
        {
            std::vector<GraphBLAS::IndexType> idx(N);
            std::vector<GraphBLAS::IndexType> idx1(N);
            for (GraphBLAS::IndexType i = 0; i < N; ++i)
            {
                idx[i] = i;
                idx1[i]= i + 1;  // off-by-one until structure only mask
            }

            index_ramp.build(idx.begin(), idx1.begin(), N);
        }

        // Set the roots parents to themselves using one-based indices because
        // the mask is sensitive to stored zeros.
        parent_list = wavefront;

        // convert all stored values to their 1-based column index
        // index_of_1based(parent_list);
        GraphBLAS::eWiseMult(parent_list,
                             GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                             GraphBLAS::First<GraphBLAS::IndexType>(),
                             index_ramp, parent_list);

        while (wavefront.nvals() > 0)
        {
            // convert all stored values to their 1-based column index
            //index_of_1based(wavefront);
            GraphBLAS::eWiseMult(wavefront,
                                 GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                                 GraphBLAS::First<GraphBLAS::IndexType>(),
                                 index_ramp, wavefront);

            // Select1st because we are left multiplying wavefront rows
            // Masking out the parent list ensures wavefront values do not
            // overlap values already stored in the parent list
            GraphBLAS::vxm(wavefront,
                           GraphBLAS::complement(parent_list),
                           GraphBLAS::NoAccumulate(),
                           GraphBLAS::MinSelect1stSemiring<T>(),
                           wavefront, graph, GraphBLAS::REPLACE);

            // We don't need to mask here since we did it in mxm.
            // Merges new parents in current wavefront with existing parents
            // parent_list<!parent_list,merge> += wavefront
            GraphBLAS::apply(parent_list,
                             GraphBLAS::NoMask(),
                             GraphBLAS::Plus<T>(),
                             GraphBLAS::Identity<T>(),
                             wavefront,
                             GraphBLAS::MERGE);
        }

        // Restore zero-based indices by subtracting 1 from all stored values
        GraphBLAS::BinaryOp_Bind2nd<unsigned int,
                                    GraphBLAS::Minus<unsigned int>>
            subtract_1(1);

        GraphBLAS::apply(parent_list,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         subtract_1,
                         parent_list,
                         GraphBLAS::REPLACE);
    }

    //************************************************************************
    /**
     * @brief Perform a set of breadth first search (BFS) traversals on the
     *        given graph.
     *
     * @param[in]  graph        N x N adjacency matrix of the graph on which to
     *                          perform a BFS. (NOT the transpose).  The value
     *                          1 should indicate an edge,
     * @param[in]  wavefronts   R x N, initial wavefront(s)/root(s) to use in the
     *                          calculation. Each row (1) corresponds to a
     *                          different BFS traversal and (2) should have a
     *                          single value set to '1' corresponding to the
     *                          root.
     * @param[out] parent_list  The list of parents for each traversal (row)
     *                          specified in the roots array.
     */
    template <typename MatrixT,
              typename WavefrontMatrixT,
              typename ParentListMatrixT>
    void bfs_batch(MatrixT const          &graph,
                   WavefrontMatrixT        wavefronts,   // copy is intentional
                   ParentListMatrixT      &parent_list)
    {
        using T = typename MatrixT::ScalarType;
        GraphBLAS::IndexType const N(graph.nrows());

        // assert parent_list is RxN
        // assert wavefront is RxN
        // assert parent_list ScalarType is GraphBLAS::IndexType

        // create index ramp for index_of() functionality
        GraphBLAS::Matrix<GraphBLAS::IndexType> index_ramp(N, N);
        {
            GraphBLAS::IndexArrayType idx, val;
            for (GraphBLAS::IndexType i = 0; i < N; ++i)
            {
                idx.push_back(i);
                val.push_back(i + 1);  // off-by-one until structure only mask
            }

            index_ramp.build(idx, idx, val);
        }

        // Set the roots parents to themselves using one-based indices because
        // the mask is sensitive to stored zeros.
        parent_list = wavefronts;
        //col_index_of_1based(parent_list);
        GraphBLAS::mxm(parent_list,
                       GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                       GraphBLAS::MinSelect2ndSemiring<T>(),
                       parent_list, index_ramp);

        while (wavefronts.nvals() > 0)
        {
            // convert all stored values to their 1-based column index
            // col_index_of_1based(wavefronts);
            GraphBLAS::mxm(wavefronts,
                           GraphBLAS::NoMask(), GraphBLAS::NoAccumulate(),
                           GraphBLAS::MinSelect2ndSemiring<T>(),
                           wavefronts, index_ramp);

            // Select1st because we are left multiplying wavefront rows
            // Masking out the parent list ensures wavefronts values do not
            // overlap values already stored in the parent list
            GraphBLAS::mxm(wavefronts,
                           GraphBLAS::complement(parent_list),
                           GraphBLAS::NoAccumulate(),
                           GraphBLAS::MinSelect1stSemiring<T>(),
                           wavefronts, graph, GraphBLAS::REPLACE);

            // We don't need to mask here since we did it in mxm.
            // Merges new parents in current wavefront with existing parents
            // parent_list<!parent_list,merge> += wavefronts
            GraphBLAS::apply(parent_list,
                             GraphBLAS::NoMask(),
                             GraphBLAS::Plus<T>(),
                             GraphBLAS::Identity<T>(),
                             wavefronts,
                             GraphBLAS::MERGE);
        }

        // Restore zero-based indices by subtracting 1 from all stored values
        GraphBLAS::BinaryOp_Bind2nd<unsigned int,
                                    GraphBLAS::Minus<unsigned int>>
            subtract_1(1);

        GraphBLAS::apply(parent_list,
                         GraphBLAS::NoMask(),
                         GraphBLAS::NoAccumulate(),
                         subtract_1,
                         parent_list,
                         GraphBLAS::REPLACE);
    }

    //************************************************************************
    /**
     * @brief Perform a single "level" breadth first search (BFS) traversal
     *        on the given graph.
     *
     * @param[in]  graph        N x N adjacency matrix of the graph on which to
     *                          perform a BFS. (NOT the transpose).  The value
     *                          1 should indicate an edge.
     * @param[in]  source       Index of the root vertex to use in the
     *                          calculation.
     * @param[out] levels       The level (distance in unweighted graphs) from
     *                          the source (root) of the BFS.
     */
    template <typename MatrixT,
              typename LevelsVectorT>
    void bfs_level(MatrixT const          &graph,
                   GraphBLAS::IndexType    source,
                   LevelsVectorT          &levels)
    {
        using T = typename MatrixT::ScalarType;
        GraphBLAS::IndexType const N(graph.nrows());

        /// @todo Assert graph is square

        GraphBLAS::Vector<bool> wavefront(N);
        wavefront.setElement(source, true);

        GraphBLAS::IndexType depth(0);
        while (wavefront.nvals() > 0)
        {
            // Increment the level
            ++depth;

            GraphBLAS::BinaryOp_Bind2nd<
                GraphBLAS::IndexType, GraphBLAS::Times<GraphBLAS::IndexType> >
                    apply_depth(depth);

            // Apply the level to all newly visited nodes
            GraphBLAS::apply(levels, GraphBLAS::NoMask(),
                             GraphBLAS::Plus<GraphBLAS::IndexType>(),
                             apply_depth, wavefront);

            GraphBLAS::mxv(wavefront, complement(levels),
                           GraphBLAS::NoAccumulate(),
                           GraphBLAS::LogicalSemiring<bool>(),
                           transpose(graph), wavefront, GraphBLAS::REPLACE);
        }
    }

    //************************************************************************
    /**
     * @brief Perform a breadth first search (BFS) on the given graph.
     *
     * @param[in]  graph        The graph to perform a BFS on.  NOT built from
     *                          the transpose of the adjacency matrix.
     *                          (1 indicates edge, structural zero = 0).
     * @param[in]  wavefront    The initial wavefront to use in the calculation.
     *                          (1 indicates root, structural zero = 0).
     * @param[out] levels       The level (distance in unweighted graphs) from
     *                          the corresponding root of that BFS
     */
    template <typename MatrixT,
              typename WavefrontMatrixT,
              typename LevelListMatrixT>
    void bfs_level(MatrixT const     &graph,
                   WavefrontMatrixT   wavefront, //row vectors, copy made
                   LevelListMatrixT  &levels)
    {
        using T = typename MatrixT::ScalarType;

        /// @todo Assert graph is square
        /// @todo Assert graph has a compatible shape with wavefront?

        GraphBLAS::IndexType rows(wavefront.nrows());
        GraphBLAS::IndexType cols(wavefront.ncols());

        unsigned int depth = 0;

        while (wavefront.nvals() > 0)
        {
            // Increment the level
            ++depth;

            // Apply the level to all newly visited nodes
            GraphBLAS::BinaryOp_Bind2nd<unsigned int,
                                        GraphBLAS::Times<unsigned int> >
                    apply_depth(depth);

            GraphBLAS::apply(levels,
                             GraphBLAS::NoMask(),
                             GraphBLAS::Plus<unsigned int>(),
                             apply_depth,
                             wavefront,
                             GraphBLAS::REPLACE);

            GraphBLAS::mxm(wavefront,
                           GraphBLAS::NoMask(),
                           GraphBLAS::NoAccumulate(),
                           GraphBLAS::LogicalSemiring<unsigned int>(),
                           wavefront, graph,
                           GraphBLAS::REPLACE);

            // Cull previously visited nodes from the wavefront
            // Replace these lines with the negate(levels) mask
            //GraphBLAS::apply(levels, not_visited,
            //                 GraphBLAS::math::IsZero<unsigned int>());
            //GraphBLAS::ewisemult(not_visited, wavefront, wavefront,
            //                     GraphBLAS::math::AndFn<unsigned int>());
            GraphBLAS::apply(
                wavefront,
                GraphBLAS::complement(levels),
                GraphBLAS::NoAccumulate(),
                GraphBLAS::Identity<typename WavefrontMatrixT::ScalarType>(),
                wavefront,
                GraphBLAS::REPLACE);
        }
    }

    //************************************************************************
    /**
     * @brief Perform a single breadth first searches (BFS) on the given graph.
     *
     * @param[in]  graph      NxN adjacency matrix of the graph on which to
     *                        perform a BFS (not the transpose).  A value of
     *                        1 indicates an edge (structural zero = 0).
     * @param[in]  wavefront  N-vector initial wavefront to use in the calculation
     *                        of R simultaneous traversals.  A value of 1 in a
     *                        given position indicates a root for the
     *                        traversal..
     * @param[out] levels     The level (distance in unweighted graphs) from
     *                        the corresponding root of that BFS.  Roots are
     *                        assigned a value of 1. (a value of 0 implies not
     *                        reachable.
     */
    template <typename MatrixT,
              typename WavefrontT,
              typename LevelListT>
    void bfs_level_masked(MatrixT const  &graph,
                          WavefrontT      wavefront, //row vector, copy made
                          LevelListT     &levels)
    {
        using T = typename MatrixT::ScalarType;

        /// Assert graph is square/have a compatible shape with wavefront
        GraphBLAS::IndexType grows(graph.nrows());
        GraphBLAS::IndexType gcols(graph.ncols());

        GraphBLAS::IndexType wsize(wavefront.size());

        if ((grows != gcols) || (wsize != grows))
        {
            throw GraphBLAS::DimensionException();
        }

        GraphBLAS::IndexType depth = 0;
        while (wavefront.nvals() > 0)
        {
            // Increment the level
            ++depth;

            // Apply the level to all newly visited nodes
            GraphBLAS::BinaryOp_Bind2nd<GraphBLAS::IndexType,
                                        GraphBLAS::Times<GraphBLAS::IndexType> >
                    apply_depth(depth);

            GraphBLAS::apply(levels,
                             GraphBLAS::NoMask(),
                             GraphBLAS::Plus<unsigned int>(),
                             apply_depth,
                             wavefront,
                             GraphBLAS::REPLACE);

            // Advance the wavefront and mask out nodes already assigned levels
            GraphBLAS::vxm(
                wavefront,
                GraphBLAS::complement(levels),
                GraphBLAS::NoAccumulate(),
                GraphBLAS::LogicalSemiring<GraphBLAS::IndexType>(),
                wavefront, graph,
                GraphBLAS::REPLACE);
        }
    }

    //************************************************************************
    /**
     * @brief Perform multiple breadth first searches (BFS) on the given graph.
     *
     * @param[in]  graph      NxN adjacency matrix of the graph on which to
     *                        perform a BFS (not the transpose).  A value of
     *                        1 indicates an edge (structural zero = 0).
     * @param[in]  wavefronts RxN initial wavefronts to use in the calculation
     *                        of R simultaneous traversals.  A value of 1 in a
     *                        given row indicates a root for the corresponding
     *                        traversal. (structural zero = 0).
     * @param[out] levels     The level (distance in unweighted graphs) from
     *                        the corresponding root of that BFS.  Roots are
     *                        assigned a value of 1. (a value of 0 implies not
     *                        reachable.
     */
    template <typename MatrixT,
              typename WavefrontsMatrixT,
              typename LevelListMatrixT>
    void batch_bfs_level_masked(MatrixT const     &graph,
                                WavefrontsMatrixT  wavefronts, //row vectors, copy
                                LevelListMatrixT  &levels)
    {
        using T = typename MatrixT::ScalarType;

        /// Assert graph is square/have a compatible shape with wavefronts
        GraphBLAS::IndexType grows(graph.nrows());
        GraphBLAS::IndexType gcols(graph.ncols());

        GraphBLAS::IndexType rows(wavefronts.nrows());
        GraphBLAS::IndexType cols(wavefronts.ncols());

        if ((grows != gcols) || (cols != grows))
        {
            throw GraphBLAS::DimensionException();
        }

        GraphBLAS::IndexType depth = 0;
        while (wavefronts.nvals() > 0)
        {
            // Increment the level
            ++depth;

            // Apply the level to all newly visited nodes
            GraphBLAS::BinaryOp_Bind2nd<GraphBLAS::IndexType,
                                        GraphBLAS::Times<GraphBLAS::IndexType> >
                    apply_depth(depth);

            GraphBLAS::apply(levels,
                             GraphBLAS::NoMask(),
                             GraphBLAS::Plus<unsigned int>(),
                             apply_depth,
                             wavefronts,
                             GraphBLAS::REPLACE);

            // Advance the wavefronts and mask out nodes already assigned levels
            GraphBLAS::mxm(
                wavefronts,
                GraphBLAS::complement(levels),
                GraphBLAS::NoAccumulate(),
                GraphBLAS::LogicalSemiring<GraphBLAS::IndexType>(),
                wavefronts, graph,
                GraphBLAS::REPLACE);
        }
    }


    //************************************************************************
    /**
     * @brief Perform a single breadth first searches (BFS) on the given graph.
     *
     * @param[in]  graph      NxN adjacency matrix of the graph on which to
     *                        perform a BFS (not the transpose).  A value of
     *                        1 indicates an edge (structural zero = 0).
     * @param[in]  wavefront  N-vector initial wavefront to use in the calculation
     *                        of R simultaneous traversals.  A value of 1 in a
     *                        given position indicates a root for the
     *                        traversal..
     * @param[out] levels     The level (distance in unweighted graphs) from
     *                        the corresponding root of that BFS.  Roots are
     *                        assigned a value of 1. (a value of 0 implies not
     *                        reachable.
     */
    template <typename MatrixT,
              typename WavefrontT,
              typename LevelListT>
    void bfs_level_masked_v2(MatrixT const  &graph,
                             WavefrontT      wavefront, //row vector, copy made
                             LevelListT     &levels)
    {
        using T = typename MatrixT::ScalarType;

        /// Assert graph is square/have a compatible shape with wavefront
        GraphBLAS::IndexType grows(graph.nrows());
        GraphBLAS::IndexType gcols(graph.ncols());

        GraphBLAS::IndexType wsize(wavefront.size());

        if ((grows != gcols) || (wsize != grows))
        {
            throw GraphBLAS::DimensionException();
        }

        GraphBLAS::IndexType depth = 0;
        while (wavefront.nvals() > 0)
        {
            // Increment the level
            ++depth;

            GraphBLAS::assign(levels,
                              wavefront,
                              GraphBLAS::NoAccumulate(),
                              depth,
                              GraphBLAS::AllIndices(),
                              GraphBLAS::MERGE);

            // Advance the wavefront and mask out nodes already assigned levels
            GraphBLAS::vxm(
                wavefront,
                GraphBLAS::complement(levels),
                GraphBLAS::NoAccumulate(),
                GraphBLAS::LogicalSemiring<GraphBLAS::IndexType>(),
                wavefront, graph,
                GraphBLAS::REPLACE);
        }
    }

}

#endif

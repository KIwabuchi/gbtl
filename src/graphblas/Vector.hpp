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
 * (https://github.com/cmu-sei/gbtl/blob/1.0.0/LICENSE) Copyright 2015 Carnegie
 * Mellon University and The Trustees of Indiana. DM17-0037, DM-0002659
 *
 * DM18-0559
 */

#pragma once

#include <cstddef>
#include <type_traits>
#include <graphblas/detail/config.hpp>
#include <graphblas/detail/param_unpack.hpp>

#define GB_INCLUDE_BACKEND_VECTOR 1
#include <backend_include.hpp>

namespace GraphBLAS
{
    template<typename ScalarT, typename... TagsT>
    class Vector;

    template<typename VectorT>
    class VectorComplementView;

    //**************************************************************************
    template<typename ScalarT, typename... TagsT>
    class Vector
    {
    public:
        typedef vector_tag          tag_type;
        typedef ScalarT ScalarType;
        typedef typename detail::vector_generator::result<
            ScalarT,
            detail::SparsenessCategoryTag,
            TagsT... ,
            detail::NullTag >::type BackendType;
        //typedef GraphBLAS::backend::Vector<ScalarT> BackendType;

        Vector() = delete;

        Vector(IndexType const & nsize) : m_vec(nsize) {}

        /**
         * @brief Construct a dense vector with 'count' copies of 'value'
         *
         * @note Calls backend constructor.
         *
         * @param[in]  count  Number of elements in the vector.
         * @param[in]  value  The scalar value to store in each element
         */
        Vector(IndexType const &count, ScalarT const &value)
            : m_vec(count, value)
        {
        }

        /**
         * @brief Construct a dense vector from dense data
         *
         * @param[in] values The dense vector from which to construct a
         *                   sparse vector from.
         *
         * @todo Should we really support this interface?
         */
        Vector(std::vector<ScalarT> const &values)
            : m_vec(values)
        {
        }

        /**
         * @brief Construct a sparse vector from dense data and a sentinel zero value.
         *
         * @param[in] values The dense vector from which to construct a
         *                   sparse vector from.
         * @param[in] zero   The "zero" value used to determine implied
         *                   zeroes (no stored value) in the sparse structure
         *
         * @todo Should we really support this interface?
         */
        Vector(std::vector<ScalarT> const &values, ScalarT zero)
            : m_vec(values, zero)
        {
        }
        /// Destructor
        ~Vector() { }

        /**
         * @brief Assignment from another vector
         *
         * @param[in]  rhs  The vector to copy from.
         *
         * @todo Should assignment work only if dimensions are same?
         * @note This clears any previous information
         */
        Vector<ScalarT, TagsT...>
        operator=(Vector<ScalarT, TagsT...> const &rhs)
        {
            if (this != &rhs)
            {
                m_vec = rhs.m_vec;
            }
            return *this;
        }

        /**
         * @brief Assignment from dense data
         *
         * @param[in]  rhs  The C++ vector of vectors to copy from.
         *
         * @todo revisit vector of vectors?
         * @todo This ignores the structural zero value.
         */
        Vector<ScalarT, TagsT...>& operator=(std::vector<ScalarT> const &rhs)
        {
            m_vec = rhs;
            return *this;
        }

        /// @todo need to change to mix and match internal types
        bool operator==(Vector<ScalarT, TagsT...> const &rhs) const
        {
            return (m_vec == rhs.m_vec);
        }

        bool operator!=(Vector<ScalarT, TagsT...> const &rhs) const
        {
            return !(*this == rhs);
        }

        /**
         * Populate the vector with stored values (using iterators).
         *
         * @param[in]  i_it      index iterator
         * @param[in]  v_it      Value (scalar) iterator
         * @param[in]  num_vals  Number of elements to store
         * @param[in]  dup       Binary function to call when value is being stored
         *                       in a location that already has a stored value.
         *                       stored_val = dup(stored_val, *v_it)
         *
         * @todo The C spec says it is an error to call build on a non-empty
         *       vector.  Unclear if the C++ should.
         */
        template<typename RAIteratorI,
                 typename RAIteratorV,
                 typename BinaryOpT = GraphBLAS::Second<ScalarType> >
        void build(RAIteratorI  i_it,
                   RAIteratorV  v_it,
                   IndexType    num_vals,
                   BinaryOpT    dup = BinaryOpT())
        {
            m_vec.build(i_it, v_it, num_vals, dup);
        }

        /**
         * Populate the vector with stored values (using iterators).
         *
         * @param[in]  indices   Array of indices
         * @param[in]  values    Array of values
         * @param[in]  dup       binary function to call when value is being stored
         *                       in a location that already has a stored value.
         *                       stored_val = dup(stored_val, *v_it)
         *
         * @todo The C spec says it is an error to call build on a non-empty
         *       vector.  Unclear if the C++ should.
         */
        template<typename ValueT,
                 typename BinaryOpT = GraphBLAS::Second<ScalarType> >
        inline void build(IndexArrayType       const &indices,
                          std::vector<ValueT>  const &values,
                          BinaryOpT                   dup = BinaryOpT())
        {
            if (indices.size() != values.size())
            {
                throw DimensionException("Vector::build");
            }
            m_vec.build(indices.begin(), values.begin(), values.size(), dup);
        }

        void clear() { m_vec.clear(); }

        IndexType size() const   { return m_vec.size(); }
        IndexType nvals() const  { return m_vec.nvals(); }

        bool hasElement(IndexType index) const
        {
            return m_vec.hasElement(index);
        }

        void setElement(IndexType index, ScalarT const &new_val)
        {
            m_vec.setElement(index, new_val);
        }

        void removeElement(IndexType index)
        {
            m_vec.removeElement(index);
        }

        /// @throw NoValueException if there is no value stored at (row,col)
        ScalarT extractElement(IndexType index) const
        {
            return m_vec.extractElement(index);
        }

        template<typename RAIteratorIT,
                 typename RAIteratorVT>
        void extractTuples(RAIteratorIT        i_it,
                           RAIteratorVT        v_it) const
        {
            m_vec.extractTuples(i_it, v_it);
        }

        void extractTuples(IndexArrayType        &indices,
                           std::vector<ScalarT>  &values) const
        {
            m_vec.extractTuples(indices, values);
        }

        /// This replaces operator<< and outputs implementation specific
        /// information.
        void printInfo(std::ostream &os) const
        {
            m_vec.printInfo(os);
        }

    private:

        // 4.3.2
        template<typename WVectorT,
                typename MaskT,
                typename AccumT,
                typename SemiringT,
                typename UVectorT,
                typename AMatrixT>
        friend inline void vxm(WVectorT         &w,
                               MaskT      const &mask,
                               AccumT     const &accum,
                               SemiringT         op,
                               UVectorT   const &u,
                               AMatrixT   const &A,
                               OutputControlEnum outp);

        // 4.3.3
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename SemiringT,
                 typename AMatrixT,
                 typename UVectorT>
        friend inline void mxv(WVectorT          &w,
                               MaskT       const &mask,
                               AccumT      const &accum,
                               SemiringT          op,
                               AMatrixT    const &A,
                               UVectorT    const &u,
                               OutputControlEnum  outp);

        // 4.3.4.1
        template<typename WScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  //can be BinaryOp, Monoid (not Semiring)
                 typename UVectorT,
                 typename VVectorT,
                 typename ...WTagsT>
        friend inline void eWiseMult(
            GraphBLAS::Vector<WScalarT, WTagsT...> &w,
            MaskT                            const &mask,
            AccumT                           const &accum,
            BinaryOpT                               op,
            UVectorT                         const &u,
            VVectorT                         const &v,
            OutputControlEnum                       outp);

        // 4.3.5.1
        template<typename WScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  //can be BinaryOp, Monoid (not Semiring)
                 typename UVectorT,
                 typename VVectorT,
                 typename ...WTagsT>
        friend inline void eWiseAdd(
            GraphBLAS::Vector<WScalarT, WTagsT...> &w,
            MaskT                            const &mask,
            AccumT                           const &accum,
            BinaryOpT                               op,
            UVectorT                         const &u,
            VVectorT                         const &v,
            OutputControlEnum                       outp);

        // 4.3.6.1
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename UVectorT,
                 typename SequenceT>
        friend inline void extract(WVectorT             &w,
                                   MaskT          const &mask,
                                   AccumT         const &accum,
                                   UVectorT       const &u,
                                   SequenceT      const &indices,
                                   OutputControlEnum     outp);

        // 4.3.6.3
        template<typename WScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename AMatrixT,
                 typename SequenceT,
                 typename ...WTags>
        friend inline void extract(
                GraphBLAS::Vector<WScalarT, WTags...> &w,
                MaskT          const &mask,
                AccumT         const &accum,
                AMatrixT       const &A,
                SequenceT      const &row_indices,
                IndexType             col_index,
                OutputControlEnum     outp);

        // 4.3.7.1: assign - standard vector variant
        // template<typename WVectorT,
        //          typename MaskT,
        //          typename AccumT,
        //          typename UVectorT,
        //          typename SequenceT,
        //          typename std::enable_if<
        //              std::is_same<vector_tag,
        //                           typename UVectorT::tag_type>::value,
        //              int>::type>
        // friend inline void assign(WVectorT           &w,
        //                           MaskT        const &mask,
        //                           AccumT       const &accum,
        //                           UVectorT     const &u,
        //                           SequenceT    const &indices,
        //                           OutputControlEnum   outp);

        template<typename WScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename UVectorT,
                 typename SequenceT,
                 typename std::enable_if<
                     std::is_same<vector_tag,
                                  typename UVectorT::tag_type>::value,
                     int>::type,
                 typename ...WTags>
        friend inline void assign(Vector<WScalarT, WTags...>   &w,
                                  MaskT        const &mask,
                                  AccumT       const &accum,
                                  UVectorT     const &u,
                                  SequenceT    const &indices,
                                  OutputControlEnum   outp);

        // 4.3.7.3:
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename UVectorT,
                 typename SequenceT,
                 typename ...CTags>
        friend inline void assign(Matrix<CScalarT, CTags...>  &C,
                                  MaskT                 const &mask,  // a vector
                                  AccumT                const &accum,
                                  UVectorT              const &u,
                                  SequenceT             const &row_indices,
                                  IndexType                    col_index,
                                  OutputControlEnum            outp);

        // 4.3.7.4:
        template<typename CScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename UVectorT,
                 typename SequenceT,
                 typename ...CTags>
        friend inline void assign(Matrix<CScalarT, CTags...>  &C,
                                  MaskT                 const &mask,  // a vector
                                  AccumT                const &accum,
                                  UVectorT              const &u,
                                  IndexType                    row_index,
                                  SequenceT             const &col_indices,
                                  OutputControlEnum            outp);

        // 4.3.7.5:
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename ValueT,
                 typename SequenceT,
                 typename std::enable_if<
                     std::is_convertible<ValueT,
                                         typename WVectorT::ScalarType>::value,
                     int>::type>
        friend inline void assign(WVectorT          &w,
                                  MaskT       const &mask,
                                  AccumT      const &accum,
                                  ValueT             val,
                                  SequenceT   const &indices,
                                  OutputControlEnum  outp);

        // 4.3.8.1: vector unaryop variant
        template<typename WScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename UnaryOpT,
                 typename UVectorT,
                 typename ...WTagsT>
        friend inline void apply(Vector<WScalarT, WTagsT...> &w,
                                 MaskT                 const &mask,
                                 AccumT                const &accum,
                                 UnaryOpT                     op,
                                 UVectorT              const &u,
                                 OutputControlEnum            outp);


        // 4.3.8.3: vector binaryop variants
        template<typename WScalarT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,
                 typename FirstT,
                 typename SecondT,
                 typename ...WTagsT>
        friend inline void apply(Vector<WScalarT, WTagsT...> &w,
                                 MaskT                 const &mask,
                                 AccumT                const &accum,
                                 BinaryOpT                    op,
                                 FirstT                const &lhs,
                                 SecondT               const &rhs,
                                 OutputControlEnum            outp);

        // 4.3.9.1
        template<typename WVectorT,
                 typename MaskT,
                 typename AccumT,
                 typename BinaryOpT,  // monoid or binary op only
                 typename AMatrixT>
        friend inline void reduce(WVectorT         &u,
                                  MaskT      const &mask,
                                  AccumT     const &accum,
                                  BinaryOpT         op,
                                  AMatrixT   const &A,
                                  OutputControlEnum outp);
        // 4.3.9.2
        template<typename ValueT,
                 typename AccumT,
                 typename MonoidT, // monoid only
                 typename UScalarT,
                 typename ...UTagsT>
        friend inline void reduce(
            ValueT                                       &dst,
            AccumT                                 const &accum,
            MonoidT                                       op,
            GraphBLAS::Vector<UScalarT, UTagsT...> const &u);

        //*********************************************************************

        template<typename OtherScalarT, typename... OtherTagsT>
        friend inline VectorComplementView<Vector<OtherScalarT,
                                                  OtherTagsT...>> complement(
            Vector<OtherScalarT, OtherTagsT...> const &mask);

        //*********************************************************************

        // .... ADD OTHER OPERATIONS AS FRIENDS AS THEY ARE IMPLEMENTED .....

    private:
        BackendType m_vec;
    };

    /**
     *  @brief Output the vector in array form.  Mainly for debugging
     *         small vectors.
     *
     *  @param[in] ostr  The output stream to send the contents
     *  @param[in] vec   The vector to output
     *  @param[in] label Optional label to output first.
     */
    template <typename VectorT>
    void print_vector(std::ostream      &ostr,
                      VectorT const     &vec,
                      std::string const &label = "")
    {
        // The new backend doesn't have get_zero.   Should we have it???
        // ostr << label << ": zero = " << vec.m_vec.get_zero() << std::endl;
        ostr << label << ":" << std::endl;
        vec.printInfo(ostr);
        ostr << std::endl;
    }

    /// @todo This does not need to be a friend
    template<typename ScalarT, typename... TagsT>
    std::ostream &operator<<(std::ostream &os, const Vector<ScalarT, TagsT...> &vec)
    {
        vec.printInfo(os);
        return os;
    }

} // end namespace GraphBLAS

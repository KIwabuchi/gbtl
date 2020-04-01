
#define GRAPHBLAS_LOGGING_LEVEL 1

#include <graphblas/graphblas.hpp>
using namespace GraphBLAS;

namespace test
{
    // 4.3.8.1: vector variant
    // template<typename WVectorT,
    //          typename MaskT,
    //          typename AccumT,
    //          typename UnaryOpT,
    //          typename UVectorT>
    // inline void apply(typename std::enable_if<
    //                      std::is_same<vector_tag,
    //                         typename WVectorT::tag_type>::value,
    //                      WVectorT>::type            &w,
    //                   //WVectorT                    &w,
    //                   MaskT                 const &mask,
    //                   AccumT                const &accum,
    //                   UnaryOpT              const &op,
    //                   UVectorT              const &u,
    //                   OutputControlEnum            outp = MERGE)
    template<typename WScalarT,
             typename MaskT,
             typename AccumT,
             typename UnaryOpT,
             typename UVectorT,
             typename ...WTagsT>
    inline void apply(Vector<WScalarT, WTagsT...> &w,
                      MaskT                 const &mask,
                      AccumT                const &accum,
                      UnaryOpT                     op,
                      UVectorT              const &u,
                      OutputControlEnum            outp = MERGE)
    {
        GRB_LOG_FN_BEGIN("apply - 4.3.8.1 - vector variant");
        GRB_LOG_FN_END("apply - 4.3.8.1 - vector variant");
    }

    // 4.3.8.2: matrix variant
    // template<typename CMatrixT,
    //          typename MaskT,
    //          typename AccumT,
    //          typename UnaryOpT,
    //          typename AMatrixT>
    // inline void apply(typename std::enable_if<
    //                      std::is_same<matrix_tag,
    //                         typename CMatrixT::tag_type>::value,
    //                      CMatrixT>::type            &C,
    //                   //CMatrixT                    &C,
    //                   MaskT                 const &Mask,
    //                   AccumT                const &accum,
    //                   UnaryOpT              const &op,
    //                   AMatrixT              const &A,
    //                   OutputControlEnum            outp = MERGE)
    template<typename CScalarT,
             typename MaskT,
             typename AccumT,
             typename UnaryOpT,
             typename AMatrixT,
             typename ...CTagsT>
    inline void apply(Matrix<CScalarT, CTagsT...> &C,
                      MaskT                 const &Mask,
                      AccumT                const &accum,
                      UnaryOpT                     op,
                      AMatrixT              const &A,
                      OutputControlEnum            outp = MERGE)
    {

        GRB_LOG_FN_BEGIN("apply - 4.3.8.2 - matrix variant");
        GRB_LOG_FN_END("apply - 4.3.8.2 - matrix variant");
    }


    // 4.3.8.3: vector binaryop bind1st variant
    // template<typename WScalarT,
    //          typename MaskT,
    //          typename AccumT,
    //          typename BinaryOpT,
    //          typename ValueT,
    //          typename UVectorT,
    //          typename std::enable_if<
    //              std::is_same<vector_tag,
    //                           typename UVectorT::tag_type>::value,
    //              int>::type,
    //          typename ...WTagsT>
    template<typename WVectorT,
             typename MaskT,
             typename AccumT,
             typename BinaryOpT,
             typename ValueT,
             typename UVectorT>
    inline void apply(WVectorT                    &w,
                      MaskT                 const &mask,
                      AccumT                const &accum,
                      BinaryOpT                    op,
                      typename std::enable_if<
                        std::is_convertible<
                          ValueT, typename BinaryOpT::first_argument_type>::value,
                        ValueT>::type            val,
                      UVectorT              const &u,
                      OutputControlEnum            outp = MERGE)
    {
        GRB_LOG_FN_BEGIN("apply - 4.3.8.3 - vector binaryop bind1st variant");
        GRB_LOG_FN_END("apply - 4.3.8.3 - vector binaryop bind1st variant");
    }

    // 4.3.8.3: vector binaryop bind2nd variant
    // template<typename WScalarT,
    //          typename MaskT,
    //          typename AccumT,
    //          typename BinaryOpT,
    //          typename UVectorT,
    //          typename ValueT,
    //          typename std::enable_if<
    //              std::is_same<vector_tag,
    //                           typename UVectorT::tag_type>::value,
    //              int>::type = 0,
    //          typename ...WTagsT>
    template<typename WVectorT,
             typename MaskT,
             typename AccumT,
             typename BinaryOpT,
             typename UVectorT,
             typename ValueT,
             typename std::enable_if<
                 std::is_same<vector_tag, typename UVectorT::tag_type>::value &&
                 std::is_same<vector_tag, typename WVectorT::tag_type>::value &&
                 std::is_convertible<
                     ValueT, typename BinaryOpT::second_argument_type>::value,
                 int>::type = 0>
    inline void apply(WVectorT                    &w,
                      MaskT                 const &mask,
                      AccumT                const &accum,
                      BinaryOpT                    op,
                      UVectorT              const &u,
                      ValueT                       val,
                      OutputControlEnum            outp = MERGE)
    {
        GRB_LOG_FN_BEGIN("apply - 4.3.8.3 - vector binaryop bind2nd variant");
        GRB_LOG_FN_END("apply - 4.3.8.3 - vector binaryop bind2nd variant");
    }

    // 4.3.8.4: matrix binaryop bind1st variant
    template<typename CScalarT,
             typename MaskT,
             typename AccumT,
             typename BinaryOpT,
             typename ValueT,
             typename AMatrixT,
             typename std::enable_if<
                 std::is_same<matrix_tag,
                              typename AMatrixT::tag_type>::value,
                 int>::type,
             typename ...CTagsT>
    inline void apply(Matrix<CScalarT, CTagsT...> &C,
                      MaskT                 const &Mask,
                      AccumT                const &accum,
                      BinaryOpT                    op,
                      ValueT                       val,
                      AMatrixT              const &A,
                      OutputControlEnum            outp = MERGE)
    {

        GRB_LOG_FN_BEGIN("apply - 4.3.8.4 - matrix binaryop bind1st variant");
        GRB_LOG_FN_END("apply - 4.3.8.4 - matrix binaryop bind2nd variant");
    }

    // 4.3.8.4: matrix binaryop bind2nd variant
    template<typename CScalarT,
             typename MaskT,
             typename AccumT,
             typename BinaryOpT,
             typename AMatrixT,
             typename ValueT,
             typename std::enable_if<
                 std::is_same<matrix_tag,
                              typename AMatrixT::tag_type>::value,
                 int>::type,
             typename ...CTagsT>
    inline void apply(Matrix<CScalarT, CTagsT...> &C,
                      MaskT                 const &Mask,
                      AccumT                const &accum,
                      BinaryOpT                    op,
                      AMatrixT              const &A,
                      ValueT                       val,
                      OutputControlEnum            outp = MERGE)
    {

        GRB_LOG_FN_BEGIN("apply - 4.3.8.4 - matrix binaryop bind2nd variant");
        GRB_LOG_FN_END("apply - 4.3.8.4 - matrix binaryop bind2nd variant");
    }

}



//****************************************************************************
int main(int argc, char **argv)
{
    Matrix<int> C(3,3), A(3,3);
    Vector<int>  w(3), u(3);
    //test::apply(w, NoMask(), NoAccumulate(), AdditiveInverse<int>(), u, REPLACE);
    //test::apply(w, NoMask(), NoAccumulate(), AdditiveInverse<int>(), u);

    //test::apply(C, NoMask(), NoAccumulate(), AdditiveInverse<int>(), A, REPLACE);
    //test::apply(C, NoMask(), NoAccumulate(), AdditiveInverse<int>(), A);

    test::apply(w, NoMask(), NoAccumulate(), Plus<int>(), (int)1, u, REPLACE);
    test::apply(w, NoMask(), NoAccumulate(), Plus<int>(), 1, u);

    test::apply(w, NoMask(), NoAccumulate(), Plus<int>(), u, 1, REPLACE);
    test::apply(w, NoMask(), NoAccumulate(), Plus<int>(), u, 1);

    //test::apply(C, NoMask(), NoAccumulate(), Plus<int>(), 1, A, REPLACE);
    //test::apply(C, NoMask(), NoAccumulate(), Plus<int>(), 1, A);

    //test::apply(C, NoMask(), NoAccumulate(), Plus<int>(), A, 1, REPLACE);
    //test::apply(C, NoMask(), NoAccumulate(), Plus<int>(), A, 1);
}

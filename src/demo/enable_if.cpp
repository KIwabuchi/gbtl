
#define GRAPHBLAS_LOGGING_LEVEL 1

#include <graphblas/graphblas.hpp>
using namespace GraphBLAS;
#if 0
namespace test
{
    // 4.3.8.1: vector variant
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


    // 4.3.8.3: vector binaryop variants
    template<typename WScalarT,
             typename MaskT,
             typename AccumT,
             typename BinaryOpT,
             typename FirstT,
             typename SecondT,
             typename ...WTagsT>
    inline void apply(Vector<WScalarT, WTagsT...> &w,
                      MaskT                 const &mask,
                      AccumT                const &accum,
                      BinaryOpT                    op,
                      FirstT                const &lhs,
                      SecondT               const &rhs,
                      OutputControlEnum            outp = MERGE)
    {
        // figure out if the user wants bind1st or bind2nd based on the argument types
        constexpr bool is_bind1st = is_vector_v<SecondT>;
        constexpr bool is_bind2nd = is_vector_v<FirstT>;

        // make sure only one of the types matches
        static_assert(is_bind1st ^ is_bind2nd, "apply isn't going to work");

        if constexpr(is_bind1st) {
            GRB_LOG_FN_BEGIN("apply - 4.3.8.3 - vector binaryop bind1st variant");
            GRB_LOG_FN_END("apply - 4.3.8.3 - vector binaryop bind1st variant");
        }
        else {
            GRB_LOG_FN_BEGIN("apply - 4.3.8.3 - vector binaryop bind2nd variant");
            GRB_LOG_FN_END("apply - 4.3.8.3 - vector binaryop bind2nd variant");
        }

    }

    // 4.3.8.4: matrix binaryop variants
    template<typename CScalarT,
             typename MaskT,
             typename AccumT,
             typename BinaryOpT,
             typename FirstT,
             typename SecondT,
             typename ...CTagsT>
    inline void apply(Matrix<CScalarT, CTagsT...> &C,
                      MaskT                 const &Mask,
                      AccumT                const &accum,
                      BinaryOpT                    op,
                      FirstT                const &lhs,
                      SecondT               const &rhs,
                      OutputControlEnum            outp = MERGE)
    {
        // figure out if the user wants bind1st or bind2nd based on the argument types
        constexpr bool is_bind1st = is_matrix_v<SecondT>;
        constexpr bool is_bind2nd = is_matrix_v<FirstT>;

        // make sure only one of the types matches
        static_assert(is_bind1st ^ is_bind2nd, "apply isn't going to work");

        if constexpr(is_bind1st) {
            GRB_LOG_FN_BEGIN("apply - 4.3.8.4 - matrix binaryop bind1st variant");
            GRB_LOG_FN_END("apply - 4.3.8.4 - matrix binaryop bind1st variant");
        }
        else {
            GRB_LOG_FN_BEGIN("apply - 4.3.8.4 - matrix binaryop bind2nd variant");
            GRB_LOG_FN_END("apply - 4.3.8.4 - matrix binaryop bind2nd variant");
        }
    }

}
#endif


//****************************************************************************
int main(int argc, char **argv)
{
    Matrix<int> C(3,3), A(3,3);
    Vector<int>  w(3), u(3);
    apply(w, NoMask(), NoAccumulate(), AdditiveInverse<int>(), u, REPLACE);
    apply(w, NoMask(), NoAccumulate(), AdditiveInverse<int>(), u);

    apply(C, NoMask(), NoAccumulate(), AdditiveInverse<int>(), A, REPLACE);
    apply(C, NoMask(), NoAccumulate(), AdditiveInverse<int>(), A);

    apply(C, NoMask(), NoAccumulate(), AdditiveInverse<int>(), transpose(A), REPLACE);
    apply(C, NoMask(), NoAccumulate(), AdditiveInverse<int>(), transpose(A));

    std::cout << "First\n";
    apply(w, NoMask(), NoAccumulate(), Plus<int>(), (int)1, u, REPLACE);
    apply(w, NoMask(), NoAccumulate(), Plus<int>(), 1, u);

    std::cout << "Second\n";
    apply(w, NoMask(), NoAccumulate(), Plus<int>(), u, 1, REPLACE);
    apply(w, NoMask(), NoAccumulate(), Plus<int>(), u, 1);

    std::cout << "First\n";
    apply(C, NoMask(), NoAccumulate(), Plus<int>(), 1, A, REPLACE);
    apply(C, NoMask(), NoAccumulate(), Plus<int>(), 1, A);

    std::cout << "Second\n";
    apply(C, NoMask(), NoAccumulate(), Plus<int>(), A, 1, REPLACE);
    apply(C, NoMask(), NoAccumulate(), Plus<int>(), A, 1);

    std::cout << "First\n";
    apply(C, NoMask(), NoAccumulate(), Plus<int>(), 1, transpose(A), REPLACE);
    apply(C, NoMask(), NoAccumulate(), Plus<int>(), 1, transpose(A));

    std::cout << "Second\n";
    apply(C, NoMask(), NoAccumulate(), Plus<int>(), transpose(A), 1, REPLACE);
    apply(C, NoMask(), NoAccumulate(), Plus<int>(), transpose(A), 1);

    std::vector<std::vector<double>> Atmp = {{1, 1, 0, 0},
                                             {1, 2, 2, 0},
                                             {0, 2, 3, 3},
                                             {0, 2, 3, 3}};
    Matrix<double, DirectedMatrixTag> AA(Atmp, 0.0);

    Matrix<uint8_t> M(4,4);
    Matrix<double, DirectedMatrixTag> CC(4, 4);
    apply(CC,
          M,
          NoAccumulate(),
          Plus<double>(),
          AA,
          0.5,
          MERGE);
}

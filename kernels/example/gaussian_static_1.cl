#define KERNEL_QUALIFIER __kernel
#define GLB_MEM_QUALIFIER __global
#define LCL_MEM_QUALIFIER __local
#define PRV_MEM_QUALIFIER __private
#define BARRIER barrier(CLK_LOCAL_MEM_FENCE);

#define ARRAY_2D(mem, size_1, idx_1, size_2, idx_2) \
                 mem[idx_1][idx_2]

#define ARRAY_2D_FLAT(mem, size_1, idx_1, size_2, idx_2) \
                      mem[(idx_1) * (size_2) + \
                                     (idx_2) ]

#if GLB_1 == 1
#define LOOP_GLB_1 /* #define glb_1 0 */
#define glb_1 0
#else
#define LOOP_GLB_1 for (size_t glb_1 = 0; glb_1 < GLB_1; ++glb_1)
#endif
#if GLB_2 == 1
#define LOOP_GLB_2 /* #define glb_2 0 */
#define glb_2 0
#else
#define LOOP_GLB_2 for (size_t glb_2 = 0; glb_2 < GLB_2; ++glb_2)
#endif

#if LCL_1 == 1
#define LOOP_LCL_1 /* #define lcl_1 0 */
#define lcl_1 0
#else
#define LOOP_LCL_1 for (size_t lcl_1 = 0; lcl_1 < LCL_1; ++lcl_1)
#endif
#if LCL_2 == 1
#define LOOP_LCL_2 /* #define lcl_2 0 */
#define lcl_2 0
#else
#define LOOP_LCL_2 for (size_t lcl_2 = 0; lcl_2 < LCL_2; ++lcl_2)
#endif

#if PRV_1 == 1
#define LOOP_PRV_1 /* #define prv_1 0 */
#define prv_1 0
#else
#define LOOP_PRV_1 for (size_t prv_1 = 0; prv_1 < PRV_1; ++prv_1)
#endif
#if PRV_2 == 1
#define LOOP_PRV_2 /* #define prv_2 0 */
#define prv_2 0
#else
#define LOOP_PRV_2 for (size_t prv_2 = 0; prv_2 < PRV_2; ++prv_2)
#endif

#if WG_1 == 1
    #define LOOP_WG_1 /* #define wg_1 0 */
    #define wg_1 0
#else
    #define LOOP_WG_1 const size_t wg_1 = get_group_id(WG_1_OCL_DIM);
#endif

#if WG_2 == 1
    #define LOOP_WG_2 /* #define wg_2 0 */
    #define wg_2 0
#else
    #define LOOP_WG_2 const size_t wg_2 = get_group_id(WG_2_OCL_DIM);
#endif

#if WI_1 == 1
    #define LOOP_WI_1 /* #define wi_1 0 */
    #define wi_1 0
#else
    #define LOOP_WI_1 const size_t wi_1 = get_local_id(WI_1_OCL_DIM);
#endif

#if WI_2 == 1
    #define LOOP_WI_2 /* #define wi_2 0 */
    #define wi_2 0
#else
    #define LOOP_WI_2 const size_t wi_2 = get_local_id(WI_2_OCL_DIM);
#endif

#define FLAT_WI_SIZE (WI_1 * WI_2)
#define FLAT_WI_IDX (wi_1 * WI_2 + \
                            wi_2 )

#define OUT_GLB(glb_1, wg_1, lcl_1, wi_1, prv_1,            \
                glb_2, wg_2, lcl_2, wi_2, prv_2)            \
        ARRAY_2D_FLAT(out_glb,                              \
                      INPUT_SIZE_1                        , \
                      glb_1 * WG_1 * LCL_1 * WI_1 * PRV_1 + \
                              wg_1 * LCL_1 * WI_1 * PRV_1 + \
                                     lcl_1 * WI_1 * PRV_1 + \
                                             wi_1 * PRV_1 + \
                                                    prv_1 , \
                      INPUT_SIZE_2                        , \
                      glb_2 * WG_2 * LCL_2 * WI_2 * PRV_2 + \
                              wg_2 * LCL_2 * WI_2 * PRV_2 + \
                                     lcl_2 * WI_2 * PRV_2 + \
                                             wi_2 * PRV_2 + \
                                                    prv_2 )

#define OUT_PRV(lcl_1, prv_1,    \
                lcl_2, prv_2)    \
        ARRAY_2D(out_prv,        \
                 LCL_1 * PRV_1 , \
                 lcl_1 * PRV_1 + \
                         prv_1 , \
                 LCL_2 * PRV_2 , \
                 lcl_2 * PRV_2 + \
                         prv_2 )

#define IN_GLB(glb_1, wg_1, lcl_1, wi_1, prv_1,                 \
               glb_2, wg_2, lcl_2, wi_2, prv_2)                 \
        ARRAY_2D_FLAT(in_glb,                                   \
                      INPUT_SIZE_1 + 4                        , \
                      glb_1  * WG_1  * LCL_1  * WI_1  * PRV_1 + \
                               wg_1  * LCL_1  * WI_1  * PRV_1 + \
                                       lcl_1  * WI_1  * PRV_1 + \
                                                wi_1  * PRV_1 + \
                                                        prv_1 , \
                      INPUT_SIZE_2 + 4                        , \
                      glb_2  * WG_2  * LCL_2  * WI_2  * PRV_2 + \
                               wg_2  * LCL_2  * WI_2  * PRV_2 + \
                                       lcl_2  * WI_2  * PRV_2 + \
                                                wi_2  * PRV_2 + \
                                                        prv_2 )

#define IN_LCL(lcl_1,  wi_1,  prv_1,          \
               lcl_2,  wi_2,  prv_2)          \
        ARRAY_2D(in_lcl,                      \
                 LCL_1  * WI_1  * PRV_1 + 4 , \
                 lcl_1  * WI_1  * PRV_1     + \
                          wi_1  * PRV_1     + \
                                  prv_1     , \
                 LCL_2  * WI_2  * PRV_2 + 4 , \
                 lcl_2  * WI_2  * PRV_2     + \
                          wi_2  * PRV_2     + \
                                  prv_2     )

#define IN_PRV(prv_1,         \
               prv_2)         \
        ARRAY_2D(in_prv,      \
                  PRV_1 + 4 , \
                  prv_1       \
                  PRV_2 + 4 , \
                  prv_2     )

KERNEL_QUALIFIER void gaussian_1(
        GLB_MEM_QUALIFIER float const * const __restrict__ in_glb,
        GLB_MEM_QUALIFIER float * const __restrict__ _,
        GLB_MEM_QUALIFIER float * const __restrict__ out_glb) {
    #if OUT_CACHE_PRV == 1
    PRV_MEM_QUALIFIER float out_prv[LCL_1 * PRV_1]
                                   [LCL_2 * PRV_2];
    #endif

    #if IN_CACHE_LCL == 1
    LCL_MEM_QUALIFIER float in_lcl[LCL_1  * WI_1  * PRV_1 + 4]
                                  [LCL_2  * WI_2  * PRV_2 + 4];
    #endif
    #if IN_CACHE_PRV == 1
    PRV_MEM_QUALIFIER float in_prv[PRV_1 + 4]
                                  [PRV_2 + 4];
    #endif

    LOOP_WG_1 {
    LOOP_WG_2 {

    LOOP_WI_1 {
    LOOP_WI_2 {

    LOOP_GLB_1 {
    LOOP_GLB_2 {

        // copy glb -> lcl
        #if IN_CACHE_LCL == 1
        {
            BARRIER
        #if 1
            #define CACHE_BLOCK_OFFSET_1 (glb_1  * WG_1  * LCL_1  * WI_1  * PRV_1  + wg_1  * LCL_1  * WI_1  * PRV_1 )
            #define CACHE_BLOCK_OFFSET_2 (glb_2  * WG_2  * LCL_2  * WI_2  * PRV_2  + wg_2  * LCL_2  * WI_2  * PRV_2 )
            #define CACHE_BLOCK_SIZE_1 (LCL_1  * WI_1  * PRV_1 + 4 )
            #define CACHE_BLOCK_SIZE_2 (LCL_2  * WI_2  * PRV_2 + 4 )
            #define CACHE_BLOCK_SIZE (CACHE_BLOCK_SIZE_1 * CACHE_BLOCK_SIZE_2)
            #if (INPUT_SIZE_2 + 4) % 4 == 0 && CACHE_BLOCK_OFFSET_2 % 4 == 0 && CACHE_BLOCK_SIZE_2 % 4 == 0
            #define VECTOR_LOAD_SIZE 4
            #define VECTOR_LOAD_TYPE float4
            #elif (INPUT_SIZE_2 + 4) % 2 == 0 && CACHE_BLOCK_OFFSET_2 % 2 == 0 && CACHE_BLOCK_SIZE_2 % 2 == 0
            #define VECTOR_LOAD_SIZE 2
            #define VECTOR_LOAD_TYPE float2
            #else
            #define VECTOR_LOAD_SIZE 1
            #define VECTOR_LOAD_TYPE float
            #endif
            #if (CACHE_BLOCK_SIZE / VECTOR_LOAD_SIZE) / FLAT_WI_SIZE > 0
            #if (CACHE_BLOCK_SIZE / VECTOR_LOAD_SIZE) / FLAT_WI_SIZE == 1
            #define step 0
            #else
            #pragma unroll
            for (size_t step = 0; step < (CACHE_BLOCK_SIZE / VECTOR_LOAD_SIZE) / FLAT_WI_SIZE; ++step)
            #endif
            {
                ARRAY_2D_FLAT(((LCL_MEM_QUALIFIER VECTOR_LOAD_TYPE*)in_lcl),
                              CACHE_BLOCK_SIZE_1,
                              (step * FLAT_WI_SIZE + FLAT_WI_IDX) / ((CACHE_BLOCK_SIZE_2 / VECTOR_LOAD_SIZE)) % (CACHE_BLOCK_SIZE_1),
                              (CACHE_BLOCK_SIZE_2 / VECTOR_LOAD_SIZE),
                              (step * FLAT_WI_SIZE + FLAT_WI_IDX) % ((CACHE_BLOCK_SIZE_2 / VECTOR_LOAD_SIZE))) =
                ARRAY_2D_FLAT(((GLB_MEM_QUALIFIER VECTOR_LOAD_TYPE*)in_glb),
                              INPUT_SIZE_1 + 4,
                              CACHE_BLOCK_OFFSET_1 + ((step * FLAT_WI_SIZE + FLAT_WI_IDX) / ((CACHE_BLOCK_SIZE_2 / VECTOR_LOAD_SIZE)) % (CACHE_BLOCK_SIZE_1)),
                              ((INPUT_SIZE_2 + 4) / VECTOR_LOAD_SIZE),
                              (CACHE_BLOCK_OFFSET_2 / VECTOR_LOAD_SIZE) + ((step * FLAT_WI_SIZE + FLAT_WI_IDX) % ((CACHE_BLOCK_SIZE_2 / VECTOR_LOAD_SIZE))));
            }
            #endif
            #if (CACHE_BLOCK_SIZE / VECTOR_LOAD_SIZE) % FLAT_WI_SIZE != 0
            if (FLAT_WI_IDX < ((CACHE_BLOCK_SIZE / VECTOR_LOAD_SIZE) % FLAT_WI_SIZE)) {
                ARRAY_2D_FLAT(((LCL_MEM_QUALIFIER VECTOR_LOAD_TYPE*)in_lcl),
                              CACHE_BLOCK_SIZE_1,
                              (((CACHE_BLOCK_SIZE / VECTOR_LOAD_SIZE) / FLAT_WI_SIZE) * FLAT_WI_SIZE + FLAT_WI_IDX) / ((CACHE_BLOCK_SIZE_2 / VECTOR_LOAD_SIZE)) % (CACHE_BLOCK_SIZE_1),
                              (CACHE_BLOCK_SIZE_2 / VECTOR_LOAD_SIZE),
                              (((CACHE_BLOCK_SIZE / VECTOR_LOAD_SIZE) / FLAT_WI_SIZE) * FLAT_WI_SIZE + FLAT_WI_IDX) % ((CACHE_BLOCK_SIZE_2 / VECTOR_LOAD_SIZE))) =
                ARRAY_2D_FLAT(((GLB_MEM_QUALIFIER VECTOR_LOAD_TYPE*)in_glb),
                              INPUT_SIZE_1 + 4,
                              CACHE_BLOCK_OFFSET_1 + ((((CACHE_BLOCK_SIZE / VECTOR_LOAD_SIZE) / FLAT_WI_SIZE) * FLAT_WI_SIZE + FLAT_WI_IDX) / ((CACHE_BLOCK_SIZE_2 / VECTOR_LOAD_SIZE)) % (CACHE_BLOCK_SIZE_1)),
                              ((INPUT_SIZE_2 + 4) / VECTOR_LOAD_SIZE),
                              (CACHE_BLOCK_OFFSET_2 / VECTOR_LOAD_SIZE) + ((((CACHE_BLOCK_SIZE / VECTOR_LOAD_SIZE) / FLAT_WI_SIZE) * FLAT_WI_SIZE + FLAT_WI_IDX) % ((CACHE_BLOCK_SIZE_2 / VECTOR_LOAD_SIZE))));
            }
            #endif

            #ifdef step
            #undef step
            #endif
            #undef VECTOR_LOAD_TYPE
            #undef VECTOR_LOAD_SIZE
            #undef CACHE_BLOCK_SIZE
            #undef CACHE_BLOCK_SIZE_2
            #undef CACHE_BLOCK_SIZE_1
            #undef CACHE_BLOCK_OFFSET_2
            #undef CACHE_BLOCK_OFFSET_1
        #else
            not implemented
        #endif
        }
        #endif
        #if IN_CACHE_LCL == 1
        BARRIER
        #endif

        LOOP_LCL_1 {
        LOOP_LCL_2 {

            // copy lcl -> prv
            #if IN_CACHE_PRV == 1
            {
                for (size_t prv_1 = 0; prv_1 < PRV_1 + 4; ++prv_1)
                for (size_t prv_2 = 0; prv_2 < PRV_2 + 4; ++prv_2)
                    IN_PRV(prv_1,
                           prv_2) =
                    #if IN_CACHE_LCL == 1
                    IN_LCL(lcl_1,  wi_1,  prv_1,
                           lcl_2,  wi_2,  prv_2)
                    #else
                    IN_GLB(glb_1,  wg_1,  lcl_1,  wi_1,  prv_1,
                           glb_2,  wg_2,  lcl_2,  wi_2,  prv_2)
                    #endif
                        ;
            }
            #endif

            LOOP_PRV_1 {
            LOOP_PRV_2 {

            // scalar phase
            #if OUT_CACHE_PRV == 1
            OUT_PRV(lcl_1, prv_1,
                    lcl_2, prv_2)
            #else
            OUT_GLB(glb_1, wg_1, lcl_1, wi_1, prv_1,
                    glb_2, wg_2, lcl_2, wi_2, prv_2)
            #endif
                =
              2.0f *
            #if IN_CACHE_PRV == 1
            IN_PRV(prv_1 + 0,
                   prv_2 + 0)
            #elif IN_CACHE_LCL == 1
            IN_LCL(lcl_1,  wi_1,  prv_1 + 0,
                   lcl_2,  wi_2,  prv_2 + 0)
            #else
            IN_GLB(glb_1,  wg_1,  lcl_1,  wi_1,  prv_1 + 0,
                   glb_2,  wg_2,  lcl_2,  wi_2,  prv_2 + 0)
            #endif
            + 4.0f *
            #if IN_CACHE_PRV == 1
            IN_PRV(prv_1 + 0,
                   prv_2 + 1)
            #elif IN_CACHE_LCL == 1
            IN_LCL(lcl_1,  wi_1,  prv_1 + 0,
                   lcl_2,  wi_2,  prv_2 + 1)
            #else
            IN_GLB(glb_1,  wg_1,  lcl_1,  wi_1,  prv_1 + 0,
                   glb_2,  wg_2,  lcl_2,  wi_2,  prv_2 + 1)
            #endif
            + 5.0f *
            #if IN_CACHE_PRV == 1
            IN_PRV(prv_1 + 0,
                   prv_2 + 2)
            #elif IN_CACHE_LCL == 1
            IN_LCL(lcl_1,  wi_1,  prv_1 + 0,
                   lcl_2,  wi_2,  prv_2 + 2)
            #else
            IN_GLB(glb_1,  wg_1,  lcl_1,  wi_1,  prv_1 + 0,
                   glb_2,  wg_2,  lcl_2,  wi_2,  prv_2 + 2)
            #endif
            + 4.0f *
            #if IN_CACHE_PRV == 1
            IN_PRV(prv_1 + 0,
                   prv_2 + 3)
            #elif IN_CACHE_LCL == 1
            IN_LCL(lcl_1,  wi_1,  prv_1 + 0,
                   lcl_2,  wi_2,  prv_2 + 3)
            #else
            IN_GLB(glb_1,  wg_1,  lcl_1,  wi_1,  prv_1 + 0,
                   glb_2,  wg_2,  lcl_2,  wi_2,  prv_2 + 3)
            #endif
            + 2.0f *
            #if IN_CACHE_PRV == 1
            IN_PRV(prv_1 + 0,
                   prv_2 + 4)
            #elif IN_CACHE_LCL == 1
            IN_LCL(lcl_1,  wi_1,  prv_1 + 0,
                   lcl_2,  wi_2,  prv_2 + 4)
            #else
            IN_GLB(glb_1,  wg_1,  lcl_1,  wi_1,  prv_1 + 0,
                   glb_2,  wg_2,  lcl_2,  wi_2,  prv_2 + 4)
            #endif

            + 4.0f *
            #if IN_CACHE_PRV == 1
            IN_PRV(prv_1 + 1,
                   prv_2 + 0)
            #elif IN_CACHE_LCL == 1
            IN_LCL(lcl_1,  wi_1,  prv_1 + 1,
                   lcl_2,  wi_2,  prv_2 + 0)
            #else
            IN_GLB(glb_1,  wg_1,  lcl_1,  wi_1,  prv_1 + 1,
                   glb_2,  wg_2,  lcl_2,  wi_2,  prv_2 + 0)
            #endif
            + 9.0f *
            #if IN_CACHE_PRV == 1
            IN_PRV(prv_1 + 1,
                   prv_2 + 1)
            #elif IN_CACHE_LCL == 1
            IN_LCL(lcl_1,  wi_1,  prv_1 + 1,
                   lcl_2,  wi_2,  prv_2 + 1)
            #else
            IN_GLB(glb_1,  wg_1,  lcl_1,  wi_1,  prv_1 + 1,
                   glb_2,  wg_2,  lcl_2,  wi_2,  prv_2 + 1)
            #endif
            + 12.0f *
            #if IN_CACHE_PRV == 1
            IN_PRV(prv_1 + 1,
                   prv_2 + 2)
            #elif IN_CACHE_LCL == 1
            IN_LCL(lcl_1,  wi_1,  prv_1 + 1,
                   lcl_2,  wi_2,  prv_2 + 2)
            #else
            IN_GLB(glb_1,  wg_1,  lcl_1,  wi_1,  prv_1 + 1,
                   glb_2,  wg_2,  lcl_2,  wi_2,  prv_2 + 2)
            #endif
            + 9.0f *
            #if IN_CACHE_PRV == 1
            IN_PRV(prv_1 + 1,
                   prv_2 + 3)
            #elif IN_CACHE_LCL == 1
            IN_LCL(lcl_1,  wi_1,  prv_1 + 1,
                   lcl_2,  wi_2,  prv_2 + 3)
            #else
            IN_GLB(glb_1,  wg_1,  lcl_1,  wi_1,  prv_1 + 1,
                   glb_2,  wg_2,  lcl_2,  wi_2,  prv_2 + 3)
            #endif
            + 4.0f *
            #if IN_CACHE_PRV == 1
            IN_PRV(prv_1 + 1,
                   prv_2 + 4)
            #elif IN_CACHE_LCL == 1
            IN_LCL(lcl_1,  wi_1,  prv_1 + 1,
                   lcl_2,  wi_2,  prv_2 + 4)
            #else
            IN_GLB(glb_1,  wg_1,  lcl_1,  wi_1,  prv_1 + 1,
                   glb_2,  wg_2,  lcl_2,  wi_2,  prv_2 + 4)
            #endif

            + 5.0f *
            #if IN_CACHE_PRV == 1
            IN_PRV(prv_1 + 2,
                   prv_2 + 0)
            #elif IN_CACHE_LCL == 1
            IN_LCL(lcl_1,  wi_1,  prv_1 + 2,
                   lcl_2,  wi_2,  prv_2 + 0)
            #else
            IN_GLB(glb_1,  wg_1,  lcl_1,  wi_1,  prv_1 + 2,
                   glb_2,  wg_2,  lcl_2,  wi_2,  prv_2 + 0)
            #endif
            + 12.0f *
            #if IN_CACHE_PRV == 1
            IN_PRV(prv_1 + 2,
                   prv_2 + 1)
            #elif IN_CACHE_LCL == 1
            IN_LCL(lcl_1,  wi_1,  prv_1 + 2,
                   lcl_2,  wi_2,  prv_2 + 1)
            #else
            IN_GLB(glb_1,  wg_1,  lcl_1,  wi_1,  prv_1 + 2,
                   glb_2,  wg_2,  lcl_2,  wi_2,  prv_2 + 1)
            #endif
            + 15.0f *
            #if IN_CACHE_PRV == 1
            IN_PRV(prv_1 + 2,
                   prv_2 + 2)
            #elif IN_CACHE_LCL == 1
            IN_LCL(lcl_1,  wi_1,  prv_1 + 2,
                   lcl_2,  wi_2,  prv_2 + 2)
            #else
            IN_GLB(glb_1,  wg_1,  lcl_1,  wi_1,  prv_1 + 2,
                   glb_2,  wg_2,  lcl_2,  wi_2,  prv_2 + 2)
            #endif
            + 12.0f *
            #if IN_CACHE_PRV == 1
            IN_PRV(prv_1 + 2,
                   prv_2 + 3)
            #elif IN_CACHE_LCL == 1
            IN_LCL(lcl_1,  wi_1,  prv_1 + 2,
                   lcl_2,  wi_2,  prv_2 + 3)
            #else
            IN_GLB(glb_1,  wg_1,  lcl_1,  wi_1,  prv_1 + 2,
                   glb_2,  wg_2,  lcl_2,  wi_2,  prv_2 + 3)
            #endif
            + 5.0f *
            #if IN_CACHE_PRV == 1
            IN_PRV(prv_1 + 2,
                   prv_2 + 4)
            #elif IN_CACHE_LCL == 1
            IN_LCL(lcl_1,  wi_1,  prv_1 + 2,
                   lcl_2,  wi_2,  prv_2 + 4)
            #else
            IN_GLB(glb_1,  wg_1,  lcl_1,  wi_1,  prv_1 + 2,
                   glb_2,  wg_2,  lcl_2,  wi_2,  prv_2 + 4)
            #endif

            + 4.0f *
            #if IN_CACHE_PRV == 1
            IN_PRV(prv_1 + 3,
                   prv_2 + 0)
            #elif IN_CACHE_LCL == 1
            IN_LCL(lcl_1,  wi_1,  prv_1 + 3,
                   lcl_2,  wi_2,  prv_2 + 0)
            #else
            IN_GLB(glb_1,  wg_1,  lcl_1,  wi_1,  prv_1 + 3,
                   glb_2,  wg_2,  lcl_2,  wi_2,  prv_2 + 0)
            #endif
            + 9.0f *
            #if IN_CACHE_PRV == 1
            IN_PRV(prv_1 + 3,
                   prv_2 + 1)
            #elif IN_CACHE_LCL == 1
            IN_LCL(lcl_1,  wi_1,  prv_1 + 3,
                   lcl_2,  wi_2,  prv_2 + 1)
            #else
            IN_GLB(glb_1,  wg_1,  lcl_1,  wi_1,  prv_1 + 3,
                   glb_2,  wg_2,  lcl_2,  wi_2,  prv_2 + 1)
            #endif
            + 12.0f *
            #if IN_CACHE_PRV == 1
            IN_PRV(prv_1 + 3,
                   prv_2 + 2)
            #elif IN_CACHE_LCL == 1
            IN_LCL(lcl_1,  wi_1,  prv_1 + 3,
                   lcl_2,  wi_2,  prv_2 + 2)
            #else
            IN_GLB(glb_1,  wg_1,  lcl_1,  wi_1,  prv_1 + 3,
                   glb_2,  wg_2,  lcl_2,  wi_2,  prv_2 + 2)
            #endif
            + 9.0f *
            #if IN_CACHE_PRV == 1
            IN_PRV(prv_1 + 3,
                   prv_2 + 3)
            #elif IN_CACHE_LCL == 1
            IN_LCL(lcl_1,  wi_1,  prv_1 + 3,
                   lcl_2,  wi_2,  prv_2 + 3)
            #else
            IN_GLB(glb_1,  wg_1,  lcl_1,  wi_1,  prv_1 + 3,
                   glb_2,  wg_2,  lcl_2,  wi_2,  prv_2 + 3)
            #endif
            + 4.0f *
            #if IN_CACHE_PRV == 1
            IN_PRV(prv_1 + 3,
                   prv_2 + 4)
            #elif IN_CACHE_LCL == 1
            IN_LCL(lcl_1,  wi_1,  prv_1 + 3,
                   lcl_2,  wi_2,  prv_2 + 4)
            #else
            IN_GLB(glb_1,  wg_1,  lcl_1,  wi_1,  prv_1 + 3,
                   glb_2,  wg_2,  lcl_2,  wi_2,  prv_2 + 4)
            #endif

            + 2.0f *
            #if IN_CACHE_PRV == 1
            IN_PRV(prv_1 + 4,
                   prv_2 + 0)
            #elif IN_CACHE_LCL == 1
            IN_LCL(lcl_1,  wi_1,  prv_1 + 4,
                   lcl_2,  wi_2,  prv_2 + 0)
            #else
            IN_GLB(glb_1,  wg_1,  lcl_1,  wi_1,  prv_1 + 4,
                   glb_2,  wg_2,  lcl_2,  wi_2,  prv_2 + 0)
            #endif
            + 4.0f *
            #if IN_CACHE_PRV == 1
            IN_PRV(prv_1 + 4,
                   prv_2 + 1)
            #elif IN_CACHE_LCL == 1
            IN_LCL(lcl_1,  wi_1,  prv_1 + 4,
                   lcl_2,  wi_2,  prv_2 + 1)
            #else
            IN_GLB(glb_1,  wg_1,  lcl_1,  wi_1,  prv_1 + 4,
                   glb_2,  wg_2,  lcl_2,  wi_2,  prv_2 + 1)
            #endif
            + 5.0f *
            #if IN_CACHE_PRV == 1
            IN_PRV(prv_1 + 4,
                   prv_2 + 2)
            #elif IN_CACHE_LCL == 1
            IN_LCL(lcl_1,  wi_1,  prv_1 + 4,
                   lcl_2,  wi_2,  prv_2 + 2)
            #else
            IN_GLB(glb_1,  wg_1,  lcl_1,  wi_1,  prv_1 + 4,
                   glb_2,  wg_2,  lcl_2,  wi_2,  prv_2 + 2)
            #endif
            + 4.0f *
            #if IN_CACHE_PRV == 1
            IN_PRV(prv_1 + 4,
                   prv_2 + 3)
            #elif IN_CACHE_LCL == 1
            IN_LCL(lcl_1,  wi_1,  prv_1 + 4,
                   lcl_2,  wi_2,  prv_2 + 3)
            #else
            IN_GLB(glb_1,  wg_1,  lcl_1,  wi_1,  prv_1 + 4,
                   glb_2,  wg_2,  lcl_2,  wi_2,  prv_2 + 3)
            #endif
            + 2.0f *
            #if IN_CACHE_PRV == 1
            IN_PRV(prv_1 + 4,
                   prv_2 + 4)
            #elif IN_CACHE_LCL == 1
            IN_LCL(lcl_1,  wi_1,  prv_1 + 4,
                   lcl_2,  wi_2,  prv_2 + 4)
            #else
            IN_GLB(glb_1,  wg_1,  lcl_1,  wi_1,  prv_1 + 4,
                   glb_2,  wg_2,  lcl_2,  wi_2,  prv_2 + 4)
            #endif
            ;

            }} // PRV
        }} // LCL

        #if OUT_CACHE_PRV == 1
        { // copy prv -> glb
        LOOP_LCL_1 {
        LOOP_LCL_2 {
        LOOP_PRV_1 {
        LOOP_PRV_2 {
            OUT_GLB(glb_1, wg_1, lcl_1, wi_1, prv_1,
                    glb_2, wg_2, lcl_2, wi_2, prv_2) =
            OUT_PRV(lcl_1, prv_1,
                    lcl_2, prv_2);
        }}}}
        }
        #endif

    }} // GLB
    }} // WI
    }} // WG
} // kernel
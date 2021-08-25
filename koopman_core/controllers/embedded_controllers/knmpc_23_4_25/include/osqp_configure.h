#ifndef OSQP_CONFIGURE_H
# define OSQP_CONFIGURE_H

# ifdef __cplusplus
extern "C" {
# endif /* ifdef __cplusplus */

/* DEBUG */
/* #undef DEBUG */

/* Operating system */
/* #undef IS_LINUX */
#define IS_MAC
/* #undef IS_WINDOWS */

/* EMBEDDED */
#define EMBEDDED (2)

/* PRINTING */
/* #undef PRINTING */

/* PROFILING */
/* #undef PROFILING */

/* CTRLC */
/* #undef CTRLC */

/* DFLOAT */
/* #undef DFLOAT */

/* DLONG */
#define DLONG

/* ENABLE_MKL_PARDISO */
/* #undef ENABLE_MKL_PARDISO */

/* MEMORY MANAGEMENT */
/* #undef OSQP_CUSTOM_MEMORY */
#ifdef OSQP_CUSTOM_MEMORY
#include ""
#endif



# ifdef __cplusplus
}
# endif /* ifdef __cplusplus */

#endif /* ifndef OSQP_CONFIGURE_H */

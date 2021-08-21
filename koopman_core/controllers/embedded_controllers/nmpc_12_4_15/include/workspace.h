#ifndef WORKSPACE_H
#define WORKSPACE_H

/*
 * This file was autogenerated by OSQP-Python on August 20, 2021 at 20:15:48.
 * 
 * This file contains the prototypes for all the workspace variables needed
 * by OSQP. The actual data is contained inside workspace.c.
 */

#include "types.h"
#include "qdldl_interface.h"

// Data structure prototypes
extern csc Pdata;
extern csc Adata;
extern c_float qdata[252];
extern c_float ldata[444];
extern c_float udata[444];
extern OSQPData data;

// Settings structure prototype
extern OSQPSettings settings;

// Scaling structure prototypes
extern c_float Dscaling[252];
extern c_float Dinvscaling[252];
extern c_float Escaling[444];
extern c_float Einvscaling[444];
extern OSQPScaling scaling;

// Prototypes for linsys_solver structure
extern csc linsys_solver_L;
extern c_float linsys_solver_Dinv[696];
extern c_int linsys_solver_P[696];
extern c_float linsys_solver_bp[696];
extern c_float linsys_solver_sol[696];
extern c_float linsys_solver_rho_inv_vec[444];
extern c_int linsys_solver_Pdiag_idx[124];
extern csc linsys_solver_KKT;
extern c_int linsys_solver_PtoKKT[124];
extern c_int linsys_solver_AtoKKT[3324];
extern c_int linsys_solver_rhotoKKT[444];
extern QDLDL_float linsys_solver_D[696];
extern QDLDL_int linsys_solver_etree[696];
extern QDLDL_int linsys_solver_Lnz[696];
extern QDLDL_int   linsys_solver_iwork[2088];
extern QDLDL_bool  linsys_solver_bwork[696];
extern QDLDL_float linsys_solver_fwork[696];
extern qdldl_solver linsys_solver;

// Prototypes for solution
extern c_float xsolution[252];
extern c_float ysolution[444];

extern OSQPSolution solution;

// Prototype for info structure
extern OSQPInfo info;

// Prototypes for the workspace
extern c_float work_rho_vec[444];
extern c_float work_rho_inv_vec[444];
extern c_int work_constr_type[444];
extern c_float work_x[252];
extern c_float work_y[444];
extern c_float work_z[444];
extern c_float work_xz_tilde[696];
extern c_float work_x_prev[252];
extern c_float work_z_prev[444];
extern c_float work_Ax[444];
extern c_float work_Px[252];
extern c_float work_Aty[252];
extern c_float work_delta_y[444];
extern c_float work_Atdelta_y[252];
extern c_float work_delta_x[252];
extern c_float work_Pdelta_x[252];
extern c_float work_Adelta_x[444];
extern c_float work_D_temp[252];
extern c_float work_D_temp_A[252];
extern c_float work_E_temp[444];

extern OSQPWorkspace workspace;

#endif // ifndef WORKSPACE_H

from .edmd import Edmd
from sklearn import linear_model
from numpy import array, concatenate, zeros, dot, linalg, eye, diag, std, divide, tile, multiply, atleast_2d, ones, zeros_like

class Keedmd(Edmd):
    def __init__(self, basis, system_dim, l1_pos=0., l1_ratio_pos=0.5, l1_vel=0., l1_ratio_vel=0.5, l1_eig=0., l1_ratio_eig=0.5, acceleration_bounds=None, override_C=True, K_p = None, K_d = None, episodic=False):
        super().__init__(basis, system_dim, l1=l1_vel, l1_ratio=l1_ratio_vel, acceleration_bounds=acceleration_bounds, override_C=override_C)
        self.episodic = episodic
        self.K_p = K_p
        self.K_d = K_d
        self.Z_std = ones((basis.Nlift + basis.n, 1))
        self.l1_pos = l1_pos
        self.l1_ratio_pos = l1_ratio_pos
        self.l1_vel = l1_vel
        self.l1_ratio_vel =  l1_ratio_vel
        self.l1_eig = l1_eig
        self.l1_ratio_eig = l1_ratio_eig

        if self.basis.Lambda is None:
            raise Exception('Basis provided is not an Koopman eigenfunction basis')

    def fit(self, X, X_d, Z, Z_dot, U, U_nom):
        self.n_lift = Z.shape[0]

        if self.l1 == 0.:
            # Solve least squares problem to find A and B for velocity terms:
            if self.episodic:
                input_vel = concatenate((Z, U-U_nom),axis=0).T
            else:
                input_vel = concatenate((Z, U), axis=0).T
            output_vel = Z_dot[int(self.n/2):self.n,:].T
            sol_vel = atleast_2d(dot(linalg.pinv(input_vel),output_vel).transpose())
            A_vel = sol_vel[:,:self.n_lift]
            B_vel = sol_vel[:,self.n_lift:]

            # Construct A matrix
            self.A = zeros((self.n_lift, self.n_lift))
            self.A[:int(self.n / 2), int(self.n / 2):self.n] = eye(int(self.n / 2))  # Known kinematics
            self.A[int(self.n/2):self.n,:] = A_vel
            self.A[self.n:,self.n:] = diag(self.basis.Lambda)

            # Solve least squares problem to find B for position terms:
            if self.episodic:
                input_pos = (U-U_nom).T
            else:
                input_pos = U.T
            output_pos = (Z_dot[:int(self.n/2),:]-dot(self.A[:int(self.n/2),:],Z)).T
            B_pos = atleast_2d(dot(linalg.pinv(input_pos),output_pos).transpose())

            # Solve least squares problem to find B for eigenfunction terms:
            U_state_fb = dot(concatenate((self.K_p, self.K_d), axis=1), X)
            input_eig = (U - U_state_fb).T
            output_eig = (Z_dot[self.n:, :] - dot(self.A[self.n:, :], Z)).T
            B_eig = atleast_2d(dot(linalg.pinv(input_eig), output_eig).transpose())

            # Construct B matrix:
            self.B = concatenate((B_pos, B_vel, B_eig), axis=0)

            if self.override_C:
                self.C = zeros((self.n,self.n_lift))
                self.C[:self.n,:self.n] = eye(self.n)
                self.C = multiply(self.C, self.Z_std.transpose())
                raise Exception('Warning: Learning of C not implemented for structured regression.')

        else:
            reg_model = linear_model.ElasticNet(alpha=self.l1, l1_ratio=self.l1_ratio, fit_intercept=False,
                                                         normalize=False, selection='random', max_iter=1e5)

            # Solve least squares problem to find A and B for velocity terms:
            if self.episodic:
                input_vel = concatenate((Z, U-U_nom), axis=0).T
            else:
                input_vel = concatenate((Z, U), axis=0).T
            output_vel = Z_dot[int(self.n / 2):self.n, :].T


            reg_model.fit(input_vel, output_vel)

            sol_vel = atleast_2d(reg_model.coef_)
            A_vel = sol_vel[:, :self.n_lift]
            B_vel = sol_vel[:, self.n_lift:]

            # Construct A matrix
            self.A = zeros((self.n_lift, self.n_lift))
            self.A[:int(self.n / 2), int(self.n / 2):self.n] = eye(int(self.n / 2))  # Known kinematics
            self.A[int(self.n / 2):self.n, :] = A_vel
            self.A[self.n:, self.n:] = diag(self.basis.Lambda)

            # Solve least squares problem to find B for position terms:
            if self.episodic:
                input_pos = (U-U_nom).T
            else:
                input_pos = U.T
            output_pos = (Z_dot[:int(self.n / 2), :] - dot(self.A[:int(self.n / 2), :], Z)).T
            reg_model.fit(input_pos, output_pos)
            B_pos = atleast_2d(reg_model.coef_)


            # Solve least squares problem to find B for eigenfunction terms:
            #input_eig = (U - U_nom).T
            U_state_fb = dot(concatenate((self.K_p, self.K_d), axis=1),X)
            input_eig = (U - U_state_fb).T
            output_eig = (Z_dot[self.n:, :] - dot(self.A[self.n:, :], Z)).T
            reg_model.fit(input_eig, output_eig)
            B_eig = atleast_2d(reg_model.coef_)

            # Construct B matrix:
            self.B = concatenate((B_pos, B_vel, B_eig), axis=0)

            if self.override_C:
                self.C = zeros((self.n, self.n_lift))
                self.C[:self.n, :self.n] = eye(self.n)
                self.C = multiply(self.C, self.Z_std.transpose())
            else:
                raise Exception('Warning: Learning of C not implemented for structured regression.')

        if not self.episodic:
            if self.K_p is None or self.K_p is None:
                raise Exception('Nominal controller gains not defined.')
            # Take nominal controller into account:
            self.A[self.n:,:self.n] -= dot(self.B[self.n:,:],concatenate((self.K_p, self.K_d), axis=1))
            #B_apnd = zeros_like(self.B)   #TODO: Revert to run modified controller adjustment
            #B_apnd[self.n:,:] = -self.B[self.n:, :]
            #self.B = concatenate((self.B,B_apnd), axis=1)

    def tune_fit(self, X, X_d, Z, Z_dot, U, U_nom, l1_ratio=array([1])):

        reg_model_cv = linear_model.MultiTaskElasticNetCV(l1_ratio=l1_ratio, fit_intercept=False,
                                            normalize=False, cv=5, n_jobs=-1, selection='random', max_iter=1e5)

        # Solve least squares problem to find A and B for velocity terms:
        if self.episodic:
            input_vel = concatenate((Z, U - U_nom), axis=0).T
        else:
            input_vel = concatenate((Z, U), axis=0).T
        output_vel = Z_dot[int(self.n / 2):self.n, :].T

        reg_model_cv.fit(input_vel, output_vel)

        sol_vel = atleast_2d(reg_model_cv.coef_)
        A_vel = sol_vel[:, :self.n_lift]
        B_vel = sol_vel[:, self.n_lift:]
        self.l1_vel = reg_model_cv.alpha_
        self.l1_ratio_vel = reg_model_cv.l1_ratio_

        # Construct A matrix
        self.A = zeros((self.n_lift, self.n_lift))
        self.A[:int(self.n / 2), int(self.n / 2):self.n] = eye(int(self.n / 2))  # Known kinematics
        self.A[int(self.n / 2):self.n, :] = A_vel
        self.A[self.n:, self.n:] = diag(self.basis.Lambda)

        # Solve least squares problem to find B for position terms:
        if self.episodic:
            input_pos = (U - U_nom).T
        else:
            input_pos = U.T
        output_pos = (Z_dot[:int(self.n / 2), :] - dot(self.A[:int(self.n / 2), :], Z)).T
        reg_model_cv.fit(input_pos, output_pos)
        B_pos = atleast_2d(reg_model_cv.coef_)
        self.l1_pos = reg_model_cv.alpha_
        self.l1_ratio_pos = reg_model_cv.l1_ratio_

        # Solve least squares problem to find B for eigenfunction terms:
        input_eig = (U - U_nom).T
        output_eig = (Z_dot[self.n:, :] - dot(self.A[self.n:, :], Z)).T
        reg_model_cv.fit(input_eig, output_eig)
        B_eig = atleast_2d(reg_model_cv.coef_)
        self.l1_eig = reg_model_cv.alpha_
        self.l1_ratio_eig = reg_model_cv.l1_ratio_

        # Construct B matrix:
        self.B = concatenate((B_pos, B_vel, B_eig), axis=0)

        if self.override_C:
            self.C = zeros((self.n, self.n_lift))
            self.C[:self.n, :self.n] = eye(self.n)
            self.C = multiply(self.C, self.Z_std.transpose())
        else:
            raise Exception('Warning: Learning of C not implemented for structured regression.')

        if not self.episodic:
            if self.K_p is None or self.K_p is None:
                raise Exception('Nominal controller gains not defined.')
            self.A[self.n:, :self.n] -= dot(self.B[self.n:, :], concatenate((self.K_p, self.K_d), axis=1))

        print('KEEDMD l1 (pos, vel, eig): ', self.l1_pos, self.l1_vel, self.l1_eig)
        print('KEEDMD l1 ratio (pos, vel, eig): ', self.l1_ratio_pos, self.l1_ratio_vel, self.l1_ratio_eig)

    def lift(self, X, X_d):
        Z = self.basis.lift(X, X_d)
        output_norm = divide(concatenate((X.T, Z),axis=1),self.Z_std.transpose())
        return output_norm

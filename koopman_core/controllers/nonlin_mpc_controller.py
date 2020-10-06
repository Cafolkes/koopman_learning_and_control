from core.controllers.controller import Controller
import numpy as np
import sys
from casadi import *
import do_mpc


class NonlinMPCController(Controller):
    """
    Class for controllers MPC.

    MPCs are solved using osqp.

    Use lifting=True to solve MPC in the lifted space
    """
    def __init__(self, q, r, xmin, xmax, umin, umax):
        sys.path.append('../../../')

        # Configure model:
        model = do_mpc.model.Model('continuous')
        xpos = model.set_variable('_x', 'xpos')
        zpos = model.set_variable('_x', 'zpos')
        theta = model.set_variable('_x', 'theta')
        dxpos = model.set_variable('_x', 'dxpos')
        dzpos = model.set_variable('_x', 'dzpos')
        dtheta = model.set_variable('_x', 'dtheta')
        u = model.set_variable('_u', 'force', (2,1))

        ddxpos = model.set_variable('_z', 'ddxpos')
        ddzpos = model.set_variable('_z', 'ddzpos')
        ddtheta = model.set_variable('_z', 'ddtheta')

        model.set_rhs('xpos', dxpos)
        model.set_rhs('zpos', dzpos)
        model.set_rhs('theta', dtheta)
        model.set_rhs('dxpos', ddxpos)
        model.set_rhs('dzpos', ddzpos)
        model.set_rhs('dtheta', ddtheta)

        mass = 2.
        inertia = 1.
        prop_arm = 0.2
        gravity = 9.81
        eul_lagrange = vertcat(
            ddxpos-(-(1 / mass) * sin(theta)*u[0] - (1 / mass) * sin(theta) * u[1]),
            ddzpos-(-gravity + (1 / mass) * cos(theta) * u[0] + (1 / mass) * cos(theta) * u[1]),
            ddtheta - (-(prop_arm / inertia) * u[0] + (prop_arm / inertia) * u[1])
        )
        model.set_alg('euler_lagrange', eul_lagrange)
        #model.set_expression(expr_name='cost', expr=sum1(q*(xpos ** 2 + zpos**2 + theta**2 + dxpos**2 + dzpos**2 + dtheta**2)))
        quad_cost = q * (xpos ** 2 + zpos ** 2 + theta**2+dxpos ** 2 + dzpos ** 2 + dtheta ** 2)
        model.set_expression(expr_name='cost', expr=quad_cost)
        model.setup()

        # Configure model predictive controller:
        self.mpc = do_mpc.controller.MPC(model)
        setup_mpc = {
            'n_horizon': 200,
            'n_robust': 0,
            'open_loop': 0,
            't_step': 0.01,
            'state_discretization': 'collocation',
            'collocation_type': 'radau',
            'collocation_deg': 2,
            'collocation_ni': 2,
            'store_full_solution': True,
            # Use MA27 linear solver in ipopt for faster calculations:
            'nlpsol_opts': {'ipopt.linear_solver': 'mumps'}
        }
        self.mpc.set_param(**setup_mpc)
        self.mpc.set_objective(mterm=model.aux['cost'], lterm=model.aux['cost'])
        self.mpc.set_rterm(force=np.array([r, r]))

        self.mpc.bounds['lower', '_x', 'xpos'] = xmin[0]
        self.mpc.bounds['lower', '_x', 'zpos'] = xmin[1]
        self.mpc.bounds['lower', '_x', 'theta'] = xmin[2]
        self.mpc.bounds['lower', '_x', 'dxpos'] = xmin[3]
        self.mpc.bounds['lower', '_x', 'dzpos'] = xmin[4]
        self.mpc.bounds['lower', '_x', 'dtheta'] = xmin[5]
        self.mpc.bounds['upper', '_x', 'xpos'] = xmax[0]
        self.mpc.bounds['upper', '_x', 'zpos'] = xmax[1]
        self.mpc.bounds['upper', '_x', 'theta'] = xmax[2]
        self.mpc.bounds['upper', '_x', 'dxpos'] = xmax[3]
        self.mpc.bounds['upper', '_x', 'dzpos'] = xmax[4]
        self.mpc.bounds['upper', '_x', 'dtheta'] = xmax[5]

        self.mpc.bounds['lower','_u','force'] = umin
        self.mpc.bounds['upper', '_u', 'force'] = umax

        self.mpc.setup()

    def eval(self, x, t):
        """eval Function to evaluate controller
        
        Arguments:
            x {numpy array [ns,]} -- state
            t {float} -- time
        
        Returns:
            control action -- numpy array [Nu,]
        """
        return self.mpc.make_step(x)

    def parse_result(self):
        x_pred = np.vstack((self.mpc.data.prediction(('_x','xpos')).T,
                            self.mpc.data.prediction(('_x','zpos')).T,
                            self.mpc.data.prediction(('_x','theta')).T,
                            self.mpc.data.prediction(('_x','dxpos')).T,
                            self.mpc.data.prediction(('_x','dzpos')).T,
                            self.mpc.data.prediction(('_x','dtheta')).T,))

        return x_pred.squeeze()

    def get_control_prediction(self):
        return self.mpc.data.prediction(('_u','force')).squeeze().T
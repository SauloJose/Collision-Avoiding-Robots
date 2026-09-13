"""
    This is a code for a S-ORCA, from Snape et al. (2010). This is a variation 
    of the method ORCA, for non-holonomic robots using a effective expanded radius.

    Last update: 13/09/2026
"""
import numpy as np 
from src.orca import * 

# ============================================================================
## Helpers
def _effective_states(positions, thetas, wheel_velocities, radii, D, L):
    """
        Convert the differential states (x, y, theta, v_l, v_r) to a effective space
        
        positions:        (N, 2)  -> (x, y)
        thetas:           (N,)    -> orientation
        wheel_velocities: (N, 2)  -> (v_l, v_r)
        radii:            (N,)    -> real radius r
        D:                float   -> shift of the effective center (= r na prática)
        L:                float

        Retorna:
            eff_pos:  (N, 2)  effective center p = (x + D cosθ, y + D sinθ)
            eff_vel:  (N, 2)  velocity of the effective center (X_dot, Y_dot)
            eff_radii:(N,)    effective radius R = r + D

    """
    # initial convertion 
    c, s = np.cos(thetas), np.sin(thetas)

    D_arr = np.asarray(D, dtype=np.float64)
    if D_arr.ndim == 0:
        D_arr = np.full_like(radii, float(D_arr))
    L_arr = np.asarray(L, dtype=np.float64)
    if L_arr.ndim == 0:
        L_arr = np.full_like(radii, float(L_arr))

    eff_pos = positions + D_arr[:, None] * np.stack([c, s], axis=1)
    eff_radii = radii + D_arr

    # Formulation of the inverse matrix of M(theta) - forward kinematics
    #   X_dot = (cosθ/2 + D sinθ/L) v_l + (cosθ/2 - D sinθ/L) v_r
    #   Y_dot = (sinθ/2 - D cosθ/L) v_l + (sinθ/2 + D cosθ/L) v_r
    # L in this code is the value of the diameter of the radius of the robot

    v_l = wheel_velocities[:, 0]
    v_r = wheel_velocities[:, 1]

    X_dot = (0.5 * c + D_arr * s / L_arr) * v_l + (0.5 * c - D_arr * s / L_arr) * v_r
    Y_dot = (0.5 * s - D_arr * c / L_arr) * v_l + (0.5 * s + D_arr * c / L_arr) * v_r
    eff_vel = np.stack([X_dot, Y_dot], axis=1)
    return eff_pos, eff_vel, eff_radii

def _cartesian_to_wheels(v_cart, thetas, D, L):
    """
     Apply M(theta)^{-1}:  convert (X_dot, Y_dot) in (v_l, v_r)
     Vetorized. L can be a scalar or (N,).
    """
    # initial convertion 
    c, s = np.cos(thetas), np.sin(thetas)
    L_arr = np.asarray(L, dtype=np.float64)
    X_dot, Y_dot = v_cart[:,0], v_cart[:,1]

    # M(theta) = [[c/2 + D s/L,  c/2 - D s/L],
    #             [s/2 - D c/L,  s/2 + D c/L]]
    # det(M) = D / L  (para D, L != 0)
    
    det = D / L_arr

    inv00 = (0.5 * s + D * c / L_arr) / det
    inv01 = -(0.5 * c - D * s / L_arr) / det
    inv10 = -(0.5 * s - D * c / L_arr) / det
    inv11 = (0.5 * c + D * s / L_arr) / det

    v_l = inv00 * X_dot + inv01 * Y_dot
    v_r = inv10 * X_dot + inv11 * Y_dot
    return np.stack([v_l, v_r], axis=1)
# ============================================================================
## Main Class
class PyNHRCA(PyORCA):
    """
    NH-ORCA (Alonso et al., 2013): ORCA applied to the effective center of a
    differential-drive robot, with a final conversion to wheel velocities.

    Interface:
        input:  positions   (N, 2)  real centers (x, y)
                thetas      (N,)    orientations (rad)
                wheel_velocities (N, 2)  current (v_l, v_r)
                goals       (N, 2)  goals ALREADY in the EFFECTIVE frame
                                    (i.e., goal_eff = goal_real - D*(cosθ, sinθ))
                radii       (N,)    real radii r
        output: wheel velocities (N, 2) -> (v_l, v_r)
    """
    def __init__(
        self,
        dt=0.1,
        v_max=1.0,
        a_max=20.5,
        t_h=1.5,
        d_max=6.0,
        max_neighbors=10,
        base_bias=0.25,
        D_scale = 1.0,          # D = D_scale * r 
        wheel_base = None,      # L
    ):
        super().__init__(
            dt=dt,
            v_max=v_max,
            a_max=a_max,
            t_h=t_h,
            d_max=d_max,
            max_neighbors=max_neighbors,
            base_bias=base_bias,
        )
        self.D_scale = float(D_scale)
        self.wheel_base = wheel_base 

    def compute_wheel_velocities(
        self,
        positions,          # (N, 2)
        thetas,             # (N,)
        wheel_velocities,   # (N, 2)  (v_l, v_r) atuais
        goals,              # (N, 2)
        radii,              # (N,)
        static_pos=None,    # (M, 2) opcional
        static_radii=None,  # (M,)   opcional
    ):
        positions = np.asarray(positions, dtype=np.float64)
        thetas = np.asarray(thetas, dtype=np.float64).reshape(-1)
        wheel_velocities = np.asarray(wheel_velocities, dtype=np.float64)
        goals = np.asarray(goals, dtype=np.float64)
        radii = np.asarray(radii, dtype=np.float64).reshape(-1)

        if np.any(radii <= 0):
            raise ValueError("radii must be > 0")
        
        D = self.D_scale * float(radii.mean())

        # L scalar or (N,)
        if self.wheel_base is not None:
            L = float(self.wheel_base)
        else:
            L = 2.0 * radii   

        # Effective states
        eff_pos, eff_vel, eff_radii = _effective_states(
            positions, thetas, wheel_velocities, radii, D, L
        )

        # Calculate ORCA with effective space
        v_cart = self.compute_velocities(
            positions=eff_pos,
            velocities=eff_vel,
            goals=goals,
            radii=eff_radii,
            static_pos=static_pos,
            static_radii=static_radii,
        )

        # (X_dot, Y_dot) => (v_l, v_r)
        u = _cartesian_to_wheels(v_cart, thetas, D, L)

        # saturate the velocity
        u = np.clip(u, -self.v_max, self.v_max)
        return u 
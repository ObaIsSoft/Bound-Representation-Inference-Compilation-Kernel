
import logging
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class OpticsAdapter:
    """
    Optics Physics Oracle.
    1. Geometric: Ray Tracing (Snell's Law) for lenses/mirrors.
    2. Laser: Gaussian Beam Propagation (Focusing/Intensity).
    """
    
    def __init__(self):
        self.name = "Optics-Physics-Solver"
        
    def run_simulation(self, params: dict) -> dict:
        """
        Run Optics Calculation.
        Params:
            - type: 'GEOMETRIC' or 'LASER'
            - [Geometric]: lens_radius, ior (index of refraction), ray_heights.
            - [Laser]: power_w, w0 (waist), wavelength_nm.
        """
        sim_type = params.get("type", "UNKNOWN").upper()
        
        if sim_type == "GEOMETRIC":
            return self._solve_geometric(params)
        elif sim_type == "LASER":
            return self._solve_laser(params)
        elif sim_type == "WAVE":
            return self._solve_wave(params)
        else:
            return {"status": "error", "message": f"Unknown optics simulation type: {sim_type}"}

    def _solve_geometric(self, params):
        """
        Geometric Ray Tracer.
        Simulates parallel rays entering a Convex Lens surface.
        Uses Vector Snell's Law: n1 (v1 x n) = n2 (v2 x n).
        """
        logger.info("[OPTICS] Tracing Rays through Lens Surface...")
        
        R = params.get("lens_radius_m", 0.5) # Radius of curvature (positive for convex entering)
        n1 = 1.0 # Air
        n2 = params.get("ior", 1.5) # Glass
        ray_heights = params.get("ray_heights_m", [0.01, 0.05]) # Ray input heights
        
        results = []
        focal_points = []
        
        for h in ray_heights:
            # 1. Intersection with Spherical Surface (Apex at z=0, Center at z=R)
            # Sphere: x^2 + (z-R)^2 = R^2
            # Ray: x = h, z is variable
            # (z-R)^2 = R^2 - h^2 -> z = R - sqrt(R^2 - h^2) (First surface hit)
            if h >= R:
                results.append({"h": h, "status": "missed_lens"})
                continue
                
            scale_z = np.sqrt(R**2 - h**2)
            z_int = R - scale_z
            
            # Intersection Point P
            P = np.array([h, z_int])
            
            # Normal Vector at P (Towards Center C=(0, R))
            # Normal is (P - C) normalized. C is at [0, R] in 2D (x, z)? No, Center is on Axis.
            # Center C = [0, R]. P = [h, z_int]. 
            # Vector CP = P - C = [h, z_int - R].
            # Normal out of surface (towards incoming light) = -CP normalized?
            # Let's say Lens Center is at z=R. Apex at z=0.
            # Surface Normal vector at intersection from Center outwards.
            # N = (P - [0, R]) normalized.
            N_vec = np.array([h, z_int - R])
            N = N_vec / np.linalg.norm(N_vec)
            # Incoming Ray Vector I = [0, 1] (Propagating +z)
            I = np.array([0, 1])
            
            # Snell's Law Vector Form
            # n1 (I x N) = n2 (T x N) -> This is complicated in 2D vector algebra.
            # Easier: Use angle form for 2D.
            # Angle of Incidence theta1 = acos(dot(-I, N))
            # Ray is coming from -z? Wait, let's assume Ray +z. I=[0, 1].
            # Normal N points towards -z side (surface curve).
            # Cos(theta1) = dot(I, -N) 
            
            cos_theta1 = np.dot(I, -N)
            theta1 = np.arccos(cos_theta1)
            
            # Snell: n1 sin(t1) = n2 sin(t2)
            sin_t2 = (n1/n2) * np.sin(theta1)
            if abs(sin_t2) > 1.0:
                results.append({"h": h, "status": "TIR"}) # Total Internal Reflection
                continue
                
            theta2 = np.arcsin(sin_t2)
            
            # Deflection angle delta = theta1 - theta2
            # New Ray Angle relative to Normal? 
            # Let's solve vector T directly:
            # T = (n1/n2)I + ( (n1/n2)cos(t1) - sqrt(1 - sin^2(t2)) )N. (Standard Formula)
            mu = n1/n2
            root = np.sqrt(1 - mu**2 * (1 - cos_theta1**2))
            # Be careful with N direction. Here N points from Center to Surface (backwards ish). 
            # Let's make N point INTO the material for the formula? No, OUT against ray.
            # Standard: N points OUT. I points IN.
            # Here I points IN (+z). N points OUT (-z direction roughly). Perfect.
            
            # T = mu * I + (mu * cos_theta1 - root) * N ?? No.
            # Vector form: T = mu * I + (mu * dot(-I, N) - sqrt(1 - mu^2(1-dot^2))) * N
            
            term = mu * np.dot(I, -N) - np.sqrt(1 - mu**2 * (1 - np.dot(I, -N)**2))
            # But wait, N here is defined from Center [0, R] to Point [h, z]. 
            # Center at z=R (positive). Point at z < R. So normal points -z.
            # Correct.
            
            T = mu * I + (mu * np.dot(I, -N) - np.sqrt(1 - mu**2 * (1 - np.dot(I, -N)**2))) * (-N) # Correction?
            # Let's stick to small angle approximation for focal length estimation in POC
            # Or simplified: P_ray = P + T * s. Solve for x=0 (Axis crossing)
            
            # Calculate crossing of Z-axis (x=0)
            # T = [Tx, Tz]. Line: X = h + Tx * s, Z = z_int + Tz * s
            # X=0 -> s = -h / Tx
            # Z_cross = z_int + Tz * (-h / Tx)
            
            # Vector Refraction (Snell's Law 3D)
            # n1 (I x N) = n2 (T x N)
            # Formula: T = eta * I + (eta * c1 - c2) * (-N) ??
            # Let's standardize: 
            # I: Incident unit vector along ray (Away from source). [0, 1] (+z).
            # N: Surface unit normal (Outward from medium 2). 
            # Center of curvature is at [0, R]. Point P is [h, z].
            # Surface is convex towards incoming light. So Center is "downstream"? 
            # Example: Center at z=R. Apex z=0. Light from -inf. Ray hits z=0.
            # Normal at Apex (0,0) points to -z (-1). Correct.
            # Normal vector: N = (P - C). Normalized.
            # P=[h, z]. C=[0, R]. N_un = [h, z-R].
            # At h=0, z=0. N_un = [0, -R]. N = [0, -1]. Correct (Points -z).
            
            # Snell Formula:
            # eta = n1 / n2
            # c1 = -dot(I, N)  (Note: Need incident I against N).
            # c2 = sqrt(1 - eta^2 * (1 - c1^2))
            # T = eta * I + (eta * c1 - c2) * N
            
            eta = n1 / n2
            c1 = -np.dot(I, N) # I=[0,1], N=[x, -z]. dot is negative?
            # if N points -z, dot(I, N) is -1. c1 = 1.
            
            # Check discriminants
            discriminant = 1 - eta**2 * (1 - c1**2)
            if discriminant < 0:
                 results.append({"h": h, "status": "TIR"})
                 continue
                 
            c2 = np.sqrt(discriminant)
            T = eta * I + (eta * c1 - c2) * N # Standard Formula
            
            # Normalize T just in case
            T = T / np.linalg.norm(T)
            
            # Setup Crossing
            Tx, Tz = T[0], T[1]
            if abs(Tx) < 1e-9:
                z_cross = float('inf')
            else:
                s_cross = -h / Tx
                z_cross = z_int + Tz * s_cross
                
            focal_points.append(z_cross)
            results.append({
                "h": h, 
                "z_intersection": z_int, 
                "z_axis_crossing": z_cross,
                "deflection_angle_deg": np.degrees(theta1 - theta2)
            })

        # Estimate Focal Length
        # Average crossing for paraxial rays (small h)
        # Paraxial approx: f = R / (n-1)
        paraxial_f = R / (n2 - n1)
        simulated_f = np.mean(focal_points) if focal_points else 0.0
        
        return {
            "status": "solved",
            "method": "Vector Ray Tracing",
            "lens_radius_m": R,
            "refractive_index": n2,
            "theoretical_focal_length_m": paraxial_f,
            "simulated_focal_length_m": simulated_f,
            "aberration_spread_m": np.std(focal_points) if focal_points else 0.0,
            "rays": results
        }

    def _solve_laser(self, params):
        """
        Laser Gaussian Beam Propagation.
        Calculates Spot Size w(z) and Intensity I(z).
        """
        logger.info("[OPTICS] Calculating Laser Focus Physics...")
        
        P = params.get("power_w", 1.0) # Watts
        w0 = params.get("waist_radius_m", 0.001) # 1mm waist inputs
        lambda_nm = params.get("wavelength_nm", 1064) # Nd:YAG IR
        lambda_m = lambda_nm * 1e-9
        
        f_lens = params.get("focal_length_m", 0.1) # focusing lens
        
        # M^2 factor (Beam Quality, 1.0 = perfect Gaussian)
        M2 = params.get("M2", 1.0) 
        
        # Physics:
        # Rayleigh Range (Input) z_R = pi * w0^2 / lambda
        z_R = (np.pi * w0**2) / (lambda_m * M2)
        
        # Focused Spot Size w_f (at focal plane)
        # Formula: w_f approx (lambda * f) / (pi * w0) for geometric focus
        # More exact: w_f = w0 / sqrt(1 + (z_R/f)^2) approx -> lambda*f / pi*w0
        
        w_f = (lambda_m * f_lens * M2) / (np.pi * w0)
        
        # Peak Intensity I = 2P / (pi * w^2)
        area = np.pi * w_f**2
        intensity = (2 * P) / area # W/m^2
        
        # Critical Intensity for Fusion (e.g. 10^14 W/cm^2)
        intensity_w_cm2 = intensity / 10000.0
        
        return {
            "status": "solved",
            "method": "Gaussian Beam Propagation",
            "input_power_w": P,
            "wavelength_nm": lambda_nm,
            "focused_spot_radius_microns": w_f * 1e6,
            "peak_intensity_w_cm2": f"{intensity_w_cm2:.2e}",
            "rayleigh_range_m": z_R,
            "fusion_ignition_check": intensity_w_cm2 > 1e14
        }

    def _solve_wave(self, params):
        """
        Wave Optics & Interference.
        - Double Slit (Young's)
        - Single Slit Diffraction
        - Diffraction Grating
        - Michelson Interferometer
        """
        setup = params.get("setup", "DOUBLE_SLIT").upper()
        lambda_nm = params.get("wavelength_nm", 532.0) # Green Laser
        lambda_m = lambda_nm * 1e-9
        
        logger.info(f"[OPTICS] Calculating Wave Interference for {setup}...")
        
        if setup == "DOUBLE_SLIT":
            # Young's Experiment
            # d sin(theta) = m * lambda (Maximize)
            d = params.get("slit_separation_m", 1e-4) # 0.1 mm
            D = params.get("screen_distance_m", 1.0)
            
            # Fringe spacing (y = lambda * D / d)
            fringe_spacing = (lambda_m * D) / d
            
            return {
                "status": "solved",
                "method": "Young's Double Slit",
                "wavelength_nm": lambda_nm,
                "slit_separation_m": d,
                "screen_distance_m": D,
                "fringe_spacing_mm": fringe_spacing * 1000.0,
                "angular_separation_rad": lambda_m / d
            }
            
        elif setup == "SINGLE_SLIT":
            # Diffraction: a sin(theta) = m * lambda (Minima)
            a = params.get("slit_width_m", 1e-5) # 10 micron
            D = params.get("screen_distance_m", 1.0)
            
            # Central maximum width (between first minima m=1 and m=-1)
            # sin(theta) = lambda / a -> y = D * tan(theta) approx D * theta
            theta_min = np.arcsin(lambda_m / a) if lambda_m <= a else np.pi/2
            central_width = 2 * D * np.tan(theta_min)
            
            return {
                "status": "solved",
                "method": "Single Slit Diffraction",
                "slit_width_m": a,
                "central_max_width_mm": central_width * 1000.0,
                "first_minima_angle_deg": np.degrees(theta_min)
            }
            
        elif setup == "GRATING":
            # Diffraction Grating: d sin(theta) = m * lambda
            lines_per_mm = params.get("lines_per_mm", 600)
            d = 1e-3 / lines_per_mm
            
            max_order = int(d / lambda_m)
            orders = []
            
            for m in range(1, max_order + 1):
                sin_theta = m * lambda_m / d
                theta = np.arcsin(sin_theta)
                orders.append({
                    "order": m,
                    "angle_deg": float(np.degrees(theta))
                })
                
            return {
                "status": "solved",
                "method": "Diffraction Grating",
                "lines_per_mm": lines_per_mm,
                "slit_spacing_um": d * 1e6,
                "max_visible_order": max_order,
                "spectral_lines": orders
            }
            
        elif setup == "INTERFEROMETER":
            # Michelson: 2d = m * lambda
            # Calculate fringe shift given movement
            d_move = params.get("mirror_move_m", 1e-6) # 1 micron
            
            # Number of fringes shifted: N = 2 * d / lambda
            fringes = (2 * d_move) / lambda_m
            
            return {
                "status": "solved",
                "method": "Michelson Interferometer",
                "mirror_movement_um": d_move * 1e6,
                "wavelength_nm": lambda_nm,
                "fringe_shift_count": float(fringes),
                "phase_shift_rad": fringes * 2 * np.pi
            }
            
        else:
            return {"status": "error", "message": f"Unknown wave setup: {setup}"}

<h3 align="center">
<img width="459" height="456" alt="PyOptFlight" src="https://github.com/user-attachments/assets/70499a36-e40e-4d48-a15b-7e2c2ccd9d3a" />
</h3>

# PyOptFlight

PyOptFlight is a Python module designed to successively solve single and multistage rocket trajectory optimization problems. It handles both in-atmosphere and vacuum operations for ascent and landing missions. PyOptFlight uses a multiple shooting approach with IPOPT for large-scale non-linear local optimization. PyOptFlight is a work in progress and struggles to converge in highly constrained cases or when using more accurate aerodynamic models. Work is being done to address these and other shortcomings.

## Avaliable Problem Constraints

Many constraints are avaliable which can be added or tighted progressively to aid in solving (think progressive tightening of a slack variable). PyOptFlight is built with repreated solving and 'warm starting' in mind. Avaliable constraints are:

- Maximum Dynamic Pressure
- Maximum Angle of Attack
- Maximum Body Angular Rates
- Minimum Thrust Before Engine Shutoff
- Maximum Throttle Rate

## Avaliable Boundary Conditions

PyOptFlight allows the use of three types of boundary conditions which can be paired together to make many different two-point boundary value problems e.g. orbit-to-landing, ascent-to-orbit, ect.

-  **Latitude/Longitude Boundary**: A specific location and altitude on the celestial body tied to the celestial body's rotation angle. The celestial body's rotation angle can be a decision variable allowing for fuel optimal launch or landing time problems.
-  **State Boundary**: Direct specification of the desired start or end state and control vectors.
-  **Keplerian Boundary**: Allows specification of orbits in terms of orbital elements. The True Anomaly can be a decision variable allowing the solver to find the optimal place in an orbit to insert or exit.

## Model Physics

PyOptFlight aims to be as accurate as it can while keeping problem size feasible. A 5-DOF system is currently used with 3 translational DOF and 2 rotational DOF (Euler Angles). Axially-symmetric rockets are assumed. Engine performance metrics, ISP and Thrust, follow simple exponential relationships like atmospheric density. Currently there is no handling of wind but the atmosphere (if present) is assumed to rotate with the celestial body. Vehicle lift and drag is computed using either Axial/Normal coefficients or Lift/Drag coefficients that are functions of mach and angle of attack. These functions can interpolate real aerodynamic data to better capture mach effects. 

## Avaliable Trajectory Intialization:

- **Linear Interpolation**: Between approximated problem boundaries.
- **Cubic Bezier Splines**: Between approximated problem boundaries.
- **Gravity Turn**: Solves a gravity turn optimization subproblem that approximates the desired problem and transforms the results appropriately into 3D space. 

## Future Work

PyOptFlight is far from done and far from fast and reliable. Future work may include:

- Addition of multiple, simplified, physics models that can be toggled like constraints.
- Experimentation with global optimization techniques
- Improved local optimization with new techniques like Successive Convexification which you can follow my progress on here: https://github.com/FlyingFaller/SCvxDemo

## Disclaimer
This software is intended for academic and research purposes only. It is based entirely on publicly available information and does not contain or reference any export-controlled technology to the best of my knowledge. This code is not licensed for reuse. Please do not use, copy, or redistribute until a license is added.

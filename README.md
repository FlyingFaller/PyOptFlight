# PyOptFlight

PyOptFlight is a Python module designed to solve single and multistage rocket trajectory optimization problems. It handles both in-atmosphere and vacuum operations for ascent and landing missions.

## Modules

- **pyoptflight**: Core functionality.
- **scripts**: Testing scripts and development utilities.
- **ksp_interface**: Tools for interfacing with Kerbal Space Program (currently not connected to the core module).

## Project Structure

```plaintext
PyOptFlight/
├── defaults/
│   ├── bodies.json
│   ├── vehicles.json
│   └── stages.json
├── mesh_images/
│   ├── Earth.jpg
│   ├── Kerbin.jpg
│   ├── Mars.jpg
│   ├── Venus.jpg
│   └── Moon.jpg
├── pyoptflight/
│   ├── __init__.py
│   ├── boundary_objects.py
│   ├── functions.py
│   ├── initialize.py
│   ├── plotting.py
│   ├── setup.py
│   └── solver.py
├── scripts/
│   └── (testing and development code)
└── ksp_interface/
    └── (KSP debugging and interface tools)
```

## Future Development
- Branching for New Solvers:
  - Fatrop Branch: A dedicated branch for a Fatrop implementation.
  - Opti Stack Branch: A branch to integrate CasADi's Opti Stack.
- Enhanced Integration: Plans to further integrate the KSP interface with the core PyOptFlight functionality.

## Dependencies
PyOptFlight is built around CasADi's NLP solver. The main branch uses an IPOPT implementation. Future branches may include support for Fatrop and CasADi's Opti Stack.
### Requires:
- Python 3.6 or later
- CasADi 3.6.7 or later

## Installation
1. Clone the repository
   ```bash
   git clone <repo-url>
   cd PyOptFlight
   ```
2. Install dependencies

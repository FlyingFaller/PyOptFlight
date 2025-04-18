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
│   ├── physics.py
│   ├── plotting.py
│   ├── setup.py
│   ├── solver.py
│   └── theory.ipynb
├── scripts/
│   └── (testing and development code)
└── ksp_interface/
    └── (KSP debugging and interface tools)
```

## Future Development
- Initialization options using global solvers such as ISRES
- Enhanced Integration: Plans to further integrate the KSP interface with the core PyOptFlight functionality.
- Add a license when appropriate

## Dependencies
PyOptFlight is built around CasADi's NLP solvers
### Requires:
- Python 3.6 or later
- CasADi 3.6.7 or later
- A bunch of other stuff TBD

## Installation
1. Clone the repository
   ```bash
   git clone <repo-url>
   cd PyOptFlight
   ```
2. Install dependencies

## Disclaimer
This software is intended for academic and research purposes only. It is based entirely on publicly available information and does not contain or reference any export-controlled technology to the best of my knowledge. This code is not licensed for reuse. Please do not use, copy, or redistribute until a license is added.

VASP inputs for prl2012.

STAGE 1: run every bulk/<XC>/<element> directory. Extract the optimized lattice parameters from CONTCAR into a copy of lattice.example.json named lattice.json. For cubic cells store a; for hcp store a and c.
STAGE 2: rerun codes/23_vasp_surface_inputs.py with --lattice-json lattice.json and a new output path. Only slabs whose STATUS says production-from-optimized-lattice may enter a surface-energy table.
STAGE 3: run every slab, verify ionic/electronic convergence, and compute sigma=(E_slab-N*E_bulk_atom)/(2*A), using matching XC, POTCAR, ENCUT and converged settings.

Q2D-GGA requires VASP >=6.3 compiled with -DUSELIBXC and a Libxc build containing GGA_X_Q2D/GGA_C_Q2D. Test those identifiers before production. POTCAR.spec is a manifest only; licensed POTCAR files are not included. Confirm the potential labels and independently converge ENCUT, bulk and slab k meshes, layer count and vacuum. The papers do not fully specify the 2020 PAW datasets or bulk k mesh, so those choices are explicit reproduction settings, not claimed verbatim paper parameters.

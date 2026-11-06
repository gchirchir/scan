SCAN_1D_2D (DC-SCAN) manuscript - Overleaf project

main.tex  manuscript                 (Overleaf: Menu -> Main document -> main.tex, then Recompile)
si.tex    Supporting Information     (Overleaf: Menu -> Main document -> si.tex,   then Recompile)
eri_note.tex  reviewer support document on the s = ln u repulsion integral (Main document -> eri_note.tex)
main.pdf, si.pdf, eri_note.pdf   the same documents compiled locally by build.sh (reference copies)
codes/    the nine standalone scripts 01-09 (python3 codes/NN_name.py), README.md, results/, figures/
figures/  all figures (scan1d2d_*.pdf)

Separate project or a subfolder of a larger Overleaf project: both work.  Overleaf compiles from the
project root, so main.tex/si.tex look for their figures in figures/ and in <folder>/figures/ for the
folder names overleaf, scan, SCAN_1D_2D, scan_1d_2d, dcscan, DC-SCAN, scan1d2d.  For another folder name
add {<folder>/figures/} to the \graphicspath line near the top of main.tex and si.tex.  When several
papers share one Overleaf project, use Logs -> Clear cached files after switching the main document.

Each .tex compiles on its own with pdfLaTeX: the bibliography is embedded (no BibTeX needed) and
references to the other document are already resolved.

Hyperlinks: citations, equations, figures and tables are links inside each PDF.  Links between the
manuscript and the SI open the other PDF at the right object when both PDFs are saved in the same
folder as main.pdf and si.pdf (Overleaf names a download after the project, so rename the two
downloads to main.pdf and si.pdf).

Generated from SCAN_1D_2D/manuscript by make_overleaf.py; do not edit here.

#!/usr/bin/env python3
# ----------------------------------------------------------------------
# Run in a terminal (after code 14):
#
#     cd SCAN_1D_2D        # (the folder that contains codes/)
#     python3 codes/15_constantin2008_figure_compare.py
#
# Needs: numpy, pandas, matplotlib and the poppler tool `pdftocairo`.  Results -> codes/results/, figures -> codes/figures/
# ----------------------------------------------------------------------
r"""Curve-by-curve comparison of code 14 with the published figures of Constantin, PRB 78, 155106 (2008).

STANDALONE SCRIPT (SCAN_1D_2D/codes): every function it uses is defined here; it reads code 14's CSV files
(data, not code) and the paper PDF  subrata/PhysRevB.78.155106.pdf.

The figures in the paper PDF are vector graphics.  `pdftocairo -svg` converts pages 3-5 to SVG; every stroked
polyline inside a figure frame is a published curve.  Pixel -> data calibration uses the frame rectangle and its
tick marks (frame corners = first/last tick values, listed in FRAMES below and checked against the tick count).
Each published curve is matched to the code-14 curve with the smallest rms difference (curves are distinct; the
assignment is printed and must agree with the paper's legend/labels).  Points within 0.2 % of the frame edges
(clipped curves) are excluded.  Resolution: one point of the printed figure is ~0.004 Ha (Fig. 1), 0.0025 Ha
(Fig. 3), 0.0066 Ha (Fig. 4), 0.0126 Ha (Fig. 5); line width ~0.25 pt.
ISTLS (Figs. 3, 4) is a many-body result not computed here; it is reported as unmatched.
Fig. 2 (jellium slab) is not reproduced.
Outputs: results/constantin2008_digitized.csv, results/constantin2008_figure_comparison.csv,
         results/tab_constantin2008_figures_rows.tex, figures/constantin2008_overlay.pdf/.png (all panels),
         figures/constantin2008_overlay_fig{1,3,4,5}.pdf/.eps/.png (one per paper figure, for the SM)
"""
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ----------------------------------------------------------------- publication style (identical copy in codes 14-17, 19)
# Modelled on the JPCL He-series figures (style() of their plot.py): serif Times/STIX text, inward ticks on all four
# sides, heavy frame, framed white legend with a black edge, bold in-panel labels instead of titles, no grid, a
# marker + line style + colour per method.  Sizes are PRINT sizes: make the figure exactly as wide as it is printed
# (PUB_W) and the fonts come out at the stated points (the JPCL 40/30/27 pt on a 14-in figure printed one column wide
# are 9.6/7.2/6.5 pt; here 10/9/7.5 pt).  The colour of a method is the same in every figure (validated palette:
# normal-vision dE >= 15 for all pairs; CVD separation is carried by the distinct marker + line style of each method).
PUB_W = {'column': 3.37, 'main_wide': 6.5, 'main_geometry': 6.6,          # revtex jcp reprint (main text)
         'sm_055': 3.6, 'sm_062': 4.05, 'sm_095': 6.2}                     # SM: article, a4, 22 mm margins
PUB = {  # method -> colour, line style, marker
    'exact':  ('#000000', '-', None),
    'LDA':    ('#6e6e6e', (0, (5, 2)), 'o'),
    'PBE':    ('#c23b4f', (0, (1, 1.3)), 's'),
    'SCAN':   ('#c47a00', (0, (4, 1.5)), '^'),
    'r2SCAN': ('#9b59b6', (0, (1, 1.3)), 'v'),
    'DC':     ('#0a8f6a', '-', 'D'),
    'DCr2':   ('#1a3f8a', (0, (5, 1.5, 1, 1.5)), 'P'),
    'A':      ('#3aa6d8', (0, (5, 1.5, 1, 1.5)), 'X'),
    'B':      ('#9b59b6', (0, (4, 1.5)), '*'),
    'C':      ('#c47a00', (0, (5, 1.5, 1, 1.5)), 'h'),
    'sheet':  ('#1f5f9e', '-', 'o'),
    'wire':   ('#0a8f6a', '-', 's'),
    'sheetF': ('#b8650a', 'none', 'o'),
    'wireF':  ('#a03fb0', 'none', 's'),
    'guide':  ('#7a7a7a', (0, (5, 2)), None),
}


def pub_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 9.5,
        'axes.labelsize': 10.5,
        'axes.titlesize': 10.5,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 7.5,
        'axes.linewidth': 1.0,
        'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
        'xtick.minor.width': 0.6, 'ytick.minor.width': 0.6,
        'xtick.major.size': 4.0, 'ytick.major.size': 4.0,
        'xtick.minor.size': 2.0, 'ytick.minor.size': 2.0,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'axes.grid': False,
        'lines.linewidth': 1.5,
        'lines.markersize': 4.5,
        'lines.markeredgewidth': 0.8,
        'legend.frameon': True, 'legend.framealpha': 1.0, 'legend.fancybox': False,
        'legend.edgecolor': 'black', 'legend.facecolor': 'white',
        'legend.handlelength': 2.4, 'legend.borderpad': 0.35, 'legend.labelspacing': 0.25,
        'legend.columnspacing': 0.9, 'patch.linewidth': 0.8,
        'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def pub_line(ax, x, y, key, label=None, marks=10, lw=None, ms=None, filled=False, **kw):
    """Plot one method in its fixed colour / line style / marker; `marks` markers spread evenly along the curve."""
    c, ls, mk = PUB[key]
    st = dict(color=c, ls=ls, lw=lw or (2.0 if key == 'exact' else 1.5), label=label)
    if mk is not None:
        st.update(marker=mk, ms=ms or 4.5, mfc=c if filled else 'white', mec=c, mew=0.9,
                  markevery=(1.0 / marks) if ls != 'none' else None)
    st.update(kw)
    return ax.plot(x, y, **st)[0]


def pub_legend(ax, **kw):
    return ax.legend(**kw)


def pub_panel(ax, text, x=0.03, y=0.96, ha='left', va='top'):
    """Bold in-panel label, e.g. '(a) quasi-2D well', in place of an axes title."""
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=10, fontweight='bold')


def pub_save(fig, path_no_ext):
    for ext in ('pdf', 'eps', 'png'):
        fig.savefig(f'{path_no_ext}.{ext}', dpi=300 if ext == 'png' else None)
    plt.close(fig)
# ----------------------------------------------------------------- end of publication style

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')
PAPER = os.path.join(ROOT, 'subrata', 'PhysRevB.78.155106.pdf')

# panel: (page, frame (x0, x1, y0_top, y1_bottom) in SVG points, x range, y range (bottom, top), n_xticks, n_yticks,
#         code-14 csv, x column, candidate columns, labels in the paper)
FRAMES = {
    'Fig. 1 (top)': (3, (356.8, 549.7, 71.9, 192.7), (0, 40), (-0.55, -0.05), 9, 11, 'constantin2008_fig1.csv', 'inv_L',
                     ['exact', 'LSDA', 'PBE', 'GGA2D_c2', 'GGA2D_c8', 'GGA2D_c16']),
    'Fig. 1 (bottom)': (3, (356.8, 549.7, 232.4, 353.2), (0, 40), (-0.55, -0.10), 9, 10, 'constantin2008_fig1.csv', 'inv_L',
                        ['exact', 'LSDA', 'PBE', 'MGGA2D_c2', 'MGGA2D_c8', 'MGGA2D_c16']),
    'Fig. 3': (4, (356.7, 551.4, 71.4, 193.3), (0, 8), (-0.40, -0.10), 9, 7, 'constantin2008_fig3.csv', 'L',
               ['LSDA', 'PBE', 'MGGA2D', 'Eq18_Eq22']),
    'Fig. 4': (4, (352.0, 551.4, 579.6, 701.5), (0, 3), (-1.2, -0.4), 7, 9, 'constantin2008_fig4.csv', 'L',
               ['LSDA', 'PBE', 'MGGA2D', 'Eq18_Eq22']),
    'Fig. 5': (5, (88.2, 284.0, 71.7, 199.1), (0, 350), (-1.6, 0.0), 8, 9, 'constantin2008_fig5.csv', 'lambda',
               ['exact', 'LSDA', 'PBE', 'MGGA2D', 'Eq18_Eq22']),
}
# extra curves of this work drawn on a panel but not matched to the paper (the paper has no DC-SCAN)
EXTRA = {'Fig. 1 (top)': ['DCSCAN', 'DCr2SCAN'], 'Fig. 1 (bottom)': ['DCSCAN', 'DCr2SCAN'],
         'Fig. 3': ['DCSCAN_xc', 'DCr2SCAN_xc'], 'Fig. 4': ['DCSCAN_xc', 'DCr2SCAN_xc']}
EXTRA_STYLE = {'DCSCAN': dict(label='DC-SCAN (this work)'),
               'DCr2SCAN': dict(label=r'DC-r$^2$SCAN (this work)'),
               'DCSCAN_xc': dict(label='DC-SCAN x + SCAN c (this work)'),
               'DCr2SCAN_xc': dict(label=r'DC-r$^2$SCAN x + r$^2$SCAN c (this work)')}
LABEL = {'exact': 'exact', 'LSDA': 'LSDA', 'PBE': 'PBE', 'GGA2D_c2': '(GGA+2D)(a)', 'GGA2D_c8': '(GGA+2D)(b)',
         'GGA2D_c16': '(GGA+2D)(c)', 'MGGA2D_c2': '(MGGA+2D)(a)', 'MGGA2D_c8': '(MGGA+2D)(b)',
         'MGGA2D_c16': '(MGGA+2D)(c)', 'MGGA2D': 'MGGA+2D', 'Eq18_Eq22': 'Eq.(18)+Eq.(22)'}


# ------------------------------------------------------------------------------------------- SVG polylines
def _matrix(tr):
    M = np.eye(3)
    for kind, args in re.findall(r'(matrix|translate|scale)\(([^)]*)\)', tr or ''):
        v = [float(x) for x in re.split(r'[ ,]+', args.strip())]
        if kind == 'matrix':
            m = np.array([[v[0], v[2], v[4]], [v[1], v[3], v[5]], [0, 0, 1]])
        elif kind == 'translate':
            m = np.array([[1, 0, v[0]], [0, 1, v[1] if len(v) > 1 else 0], [0, 0, 1]])
        else:
            m = np.array([[v[0], 0, 0], [0, v[1] if len(v) > 1 else v[0], 0], [0, 0, 1]])
        M = M @ m
    return M


def stroked_polylines(svg_file):
    """All stroked sub-paths (end points of M/L/C segments) in page coordinates, with their style string."""
    out = []

    def walk(el, M):
        M2 = M @ _matrix(el.get('transform'))
        if el.tag.split('}')[-1] == 'path':
            st = (el.get('style') or '') + ' ' + ' '.join(
                f'{k}:{v}' for k, v in el.attrib.items() if k in ('stroke', 'stroke-dasharray'))
            if 'stroke:' in st and 'stroke:none' not in st.replace(' ', ''):
                toks = re.findall(r'[MLCZmlcz]|-?\d*\.?\d+(?:e-?\d+)?', el.get('d'))
                subs, sub, nums, cmd = [], [], [], None
                for tk in toks:
                    if tk in 'MLCZmlcz':
                        cmd = tk
                        if tk in 'Mm' and sub:
                            subs.append(sub)
                            sub = []
                        continue
                    nums.append(float(tk))
                    if len(nums) == {'M': 2, 'L': 2, 'C': 6}.get(cmd, 2):
                        sub.append((nums[-2], nums[-1]))
                        nums = []
                if sub:
                    subs.append(sub)
                for s in subs:
                    a = np.c_[np.array(s), np.ones(len(s))] @ M2.T
                    out.append((a[:, :2], st))
        for ch in el:
            walk(ch, M2)

    walk(ET.parse(svg_file).getroot(), np.eye(3))
    return out


def count_ticks(polys, frame):
    x0, x1, y0, y1 = frame
    yt = {round(a[0, 1], 1) for a, _ in polys if len(a) == 2 and abs(a[0, 0] - x0) < 0.6
          and abs(a[0, 1] - a[1, 1]) < 0.01 and y0 - 0.5 <= a[0, 1] <= y1 + 0.5}
    xt = {round(a[0, 0], 1) for a, _ in polys if len(a) == 2 and abs(a[0, 1] - y1) < 0.6
          and abs(a[0, 0] - a[1, 0]) < 0.01 and x0 - 0.5 <= a[0, 0] <= x1 + 0.5}
    return len(xt), len(yt)


def digitize(polys, frame, xr, yr, minpts=8):
    x0, x1, y0, y1 = frame
    out = []
    for a, st in polys:
        if len(a) < minpts:
            continue
        if not (x0 - 1 <= a[:, 0].min() and a[:, 0].max() <= x1 + 1 and y0 - 1 <= a[:, 1].min() and a[:, 1].max() <= y1 + 1):
            continue
        X = xr[0] + (a[:, 0] - x0) / (x1 - x0) * (xr[1] - xr[0])
        Y = yr[1] - (a[:, 1] - y0) / (y1 - y0) * (yr[1] - yr[0])
        dash = st.split('stroke-dasharray:')[-1].strip().split(';')[0][:24] if 'dasharray' in st else 'solid'
        o = np.argsort(X)
        out.append((dash, X[o], Y[o]))
    return out


XLABEL = {'inv_L': r'$1/L=\lambda/L_{\max}$ (bohr$^{-1}$)', 'L': r'$L=L_{\max}/\lambda$ (bohr)', 'lambda': r'$\lambda$'}
YLABEL = {'Fig. 1 (top)': r'$\epsilon_x$ (hartree)', 'Fig. 1 (bottom)': r'$\epsilon_x$ (hartree)',
          'Fig. 3': r'$\epsilon_{xc}$ (hartree)', 'Fig. 4': r'$\epsilon_{xc}$ (hartree)', 'Fig. 5': r'$E_{xc}$ (hartree)'}
NOTE = {'Fig. 1 (top)': r'$r_s^{2D}=4$, exchange', 'Fig. 1 (bottom)': r'$r_s^{2D}=4$, exchange',
        'Fig. 3': r'$r_s^{2D}=4$', 'Fig. 4': r'$r_s^{2D}=2/\sqrt{3}$', 'Fig. 5': 'scaled H atom'}


KEY = {'exact': 'exact', 'LSDA': 'LDA', 'PBE': 'PBE', 'GGA2D_c2': 'A', 'GGA2D_c8': 'B', 'GGA2D_c16': 'C',
       'MGGA2D_c2': 'A', 'MGGA2D_c8': 'B', 'MGGA2D_c16': 'C', 'MGGA2D': 'A', 'Eq18_Eq22': 'B',
       'DCSCAN': 'DC', 'DCr2SCAN': 'DCr2', 'DCSCAN_xc': 'DC', 'DCr2SCAN_xc': 'DCr2'}


def draw_panel(ax, panel, data, legend=True, legend_loc='lower right'):
    """Lines: this work (fixed colour/line style per method).  Open markers: the same curve digitized from the
    paper, in the method's colour and marker shape."""
    ours, xcol, cols, xr, yr, matched, unmatched = data
    for c in cols:
        k = KEY[c]
        pub_line(ax, ours[xcol].values, ours[c].values, k, marker='',
                 label='exact reference (this work)' if c == 'exact' else LABEL[c], zorder=3 if c == 'exact' else 2)
    for c in EXTRA.get(panel, []):
        if c in ours:
            pub_line(ax, ours[xcol].values, ours[c].values, KEY[c], marker='', lw=1.8, zorder=4,
                     label=EXTRA_STYLE[c]['label'])
    for c, X, Y in matched:
        col, _, mk = PUB[KEY[c]]
        step = max(1, len(X) // 14)                    # thin the markers for legibility
        ax.plot(X[::step], Y[::step], ls='none', marker=mk or 'o', ms=4.2, mfc='none', mew=0.9,
                color=col if KEY[c] != 'exact' else '0.25', zorder=5)
    ax.plot([], [], ls='none', marker='o', ms=4.2, mfc='none', mew=0.9, color='0.25',
            label='Constantin 2008, digitized')
    for name, X, Y in unmatched:
        ax.plot(X, Y, ls='none', marker='s', ms=2.6, color='0.45', label=name.replace(' (not computed)', ', paper only'))
    ax.set_xlim(*xr)
    ax.set_ylim(*yr)
    ax.set_xlabel(XLABEL[xcol])
    ax.set_ylabel(YLABEL[panel])
    left = panel in ('Fig. 3', 'Fig. 4')               # top-left is empty there
    ax.text(0.04 if left else 0.96, 0.29 if panel.startswith('Fig. 1') else 0.95, NOTE[panel], transform=ax.transAxes, ha='left' if left else 'right',
            va='top', fontsize=10)
    if legend:
        pub_legend(ax, loc=legend_loc, fontsize=7)


def single_figures(panels):
    """One overlay figure per paper figure (Fig. 1 as two stacked panels, as in the paper), at SM print size."""
    pub_style()
    fig, axs = plt.subplots(2, 1, figsize=(PUB_W['sm_055'], 5.4), sharex=True, gridspec_kw=dict(hspace=0.06))
    draw_panel(axs[0], 'Fig. 1 (top)', panels['Fig. 1 (top)'], legend=False)
    draw_panel(axs[1], 'Fig. 1 (bottom)', panels['Fig. 1 (bottom)'], legend=False)
    axs[0].set_xlabel('')
    pub_panel(axs[0], '(a) GGA+2D', x=0.96, y=0.40, ha='right', va='center')
    pub_panel(axs[1], '(b) MGGA+2D', x=0.96, y=0.40, ha='right', va='center')
    h, l = axs[0].get_legend_handles_labels()            # one shared legend; (a)-(c) = c = 2, 8, 16 in both panels
    rename = {'(GGA+2D)(a)': r'+2D (a), $c=2$', '(GGA+2D)(b)': r'+2D (b), $c=8$', '(GGA+2D)(c)': r'+2D (c), $c=16$'}
    axs[0].legend(h, [rename.get(x, x) for x in l], loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2,
               fontsize=7, handlelength=2.6)
    pub_save(fig, os.path.join(FIG, 'constantin2008_overlay_fig1'))
    for panel, tag in (('Fig. 3', 'fig3'), ('Fig. 4', 'fig4'), ('Fig. 5', 'fig5')):
        fig, ax = plt.subplots(figsize=(PUB_W['sm_062'], 3.0))
        draw_panel(ax, panel, panels[panel], legend_loc='center right' if panel == 'Fig. 5' else 'lower right')
        pub_save(fig, os.path.join(FIG, f'constantin2008_overlay_{tag}'))
    plt.rcdefaults()


def main():
    if not os.path.exists(PAPER):
        raise SystemExit(f'paper PDF not found: {PAPER}')
    if shutil.which('pdftocairo') is None:
        raise SystemExit('pdftocairo (poppler) not found')
    tmp = tempfile.mkdtemp()
    polys = {}
    for page in (3, 4, 5):
        svg = os.path.join(tmp, f'p{page}.svg')
        subprocess.run(['pdftocairo', '-svg', '-f', str(page), '-l', str(page), PAPER, svg], check=True)
        polys[page] = stroked_polylines(svg)

    dig_rows, cmp_rows, log = [], [], []
    panels = {}                                       # panel -> (ours, xcol, cols, xr, yr, matched, unmatched)
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (panel, (page, frame, xr, yr, nxt, nyt, csv, xcol, cols)) in zip(axs.ravel(), FRAMES.items()):
        matched, unmatched = [], []
        nx, ny = count_ticks(polys[page], frame)
        if (nx, ny) != (nxt, nyt):
            raise RuntimeError(f'{panel}: tick count {nx},{ny} != expected {nxt},{nyt} (calibration)')
        ours = pd.read_csv(os.path.join(RES, csv))
        xo = ours[xcol].values
        for c in cols:
            ax.plot(xo, ours[c].values, '-', lw=1.0, label=f'ours: {LABEL[c]}')
        for dash, X, Y in digitize(polys[page], frame, xr, yr):
            span = yr[1] - yr[0]
            m = (Y > yr[0] + 0.002 * span) & (Y < yr[1] - 0.002 * span) & (X >= xo.min()) & (X <= xo.max())
            best = None
            for c in cols:
                if m.sum() < 3:
                    break
                d = Y[m] - np.interp(X[m], xo, ours[c].values)
                r = float(np.sqrt(np.mean(d * d)))
                if best is None or r < best[1]:
                    best = (c, r, float(np.max(np.abs(d))), int(m.sum()))
            # unmatched: an rms above 5 printed-point units means the curve is not one of ours (ISTLS)
            unit = span / (frame[3] - frame[2])
            if best is None or best[1] > 5.0 * unit:
                name = 'ISTLS (not computed)' if panel in ('Fig. 3', 'Fig. 4') else 'unmatched'
                cmp_rows.append([panel, name, dash, np.nan, np.nan, int(m.sum()), unit])
                ax.plot(X, Y, 'k.', ms=2)
                unmatched.append((name, X, Y))
            else:
                cmp_rows.append([panel, LABEL[best[0]], dash, best[1], best[2], best[3], unit])
                ax.plot(X[m], Y[m], 'o', ms=2.2, mfc='none', mew=0.6, color='k')
                name = LABEL[best[0]]
                matched.append((best[0], X[m], Y[m]))
            for xv, yv in zip(X, Y):
                dig_rows.append([panel, name, dash, xv, yv])
        panels[panel] = (ours, xcol, cols, xr, yr, matched, unmatched)
        ax.set_xlim(*xr)
        ax.set_ylim(*yr)
        ax.set_title(panel + ': lines = code 14, circles = digitized paper', fontsize=9)
        ax.legend(fontsize=6, loc='lower left')
    axs.ravel()[-1].axis('off')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG, 'constantin2008_overlay.pdf'))
    fig.savefig(os.path.join(FIG, 'constantin2008_overlay.png'), dpi=140)
    plt.close(fig)
    single_figures(panels)

    pd.DataFrame(dig_rows, columns=['panel', 'curve', 'dash', 'x', 'y']).to_csv(
        os.path.join(RES, 'constantin2008_digitized.csv'), index=False)
    cmp = pd.DataFrame(cmp_rows, columns=['panel', 'curve', 'dash', 'rms_Ha', 'max_abs_Ha', 'n_points', 'Ha_per_pt'])
    cmp.to_csv(os.path.join(RES, 'constantin2008_figure_comparison.csv'), index=False)
    tex = ['% generated by codes/15_constantin2008_figure_compare.py: panel & curve & rms (mHa) & max (mHa) & points & mHa per printed pt']
    for r in cmp.itertuples():
        if np.isnan(r.rms_Ha):
            tex.append(f'{r.panel} & {r.curve} & -- & -- & -- & {1e3 * r.Ha_per_pt:.1f} \\\\')
        else:
            tex.append(f'{r.panel} & {r.curve} & {1e3 * r.rms_Ha:.1f} & {1e3 * r.max_abs_Ha:.1f} & {r.n_points} & '
                       f'{1e3 * r.Ha_per_pt:.1f} \\\\')
    with open(os.path.join(RES, 'tab_constantin2008_figures_rows.tex'), 'w') as fh:
        fh.write('\n'.join(tex) + '\n')
    print(cmp.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()

"""Figures and rejection table of chapter 3 (training on the simulated samples).
Input : analysis/ch3_data/*.json (saved ROC curves, training curves and AUCs)
Output: report/figs/ch3_*.pdf and report/ch3_rej.tex
Run   : python3 analysis/ch3_figs.py   (from the repository root)
"""
import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'report', 'figs')
P1 = json.load(open(os.path.join(HERE, 'ch3_data', 'pack_first_version.json')))['runs']
P2 = json.load(open(os.path.join(HERE, 'ch3_data', 'pack_qgnnplus.json')))
V = json.load(open(os.path.join(HERE, 'ch3_data', 'vqc_event_roc.json')))
RB = np.load(os.path.join(HERE, 'ch3_data', 'event', 'roc_bands.npz'))   # BDT ROC, bootstrap 68% band over test events (event_boot.py)
BK = {'base8': '8 jet features (no score)', 'base8+msd0': '8 features + msd0 (no score)', 'base8+q': '+ QGNN+', 'base8+msd0+q': '+ msd0 + QGNN+'}
G2 = RB['G']
# In the saved file 'sub_base8' and 'sub_msd0' average ALL runs, with and without the tagger score.
# Recover the no-score mean ROC: base8 = 3 runs with score + 3 without; msd0 = 2 with + 3 without.
def _no_score(k, n_all, n_s):
    # saved 'mean'/'std' are over all runs (population std); the with-score group is stored separately,
    # so the no-score group mean and std follow from the sums of squares.
    A = np.array(V[k + '_s0']['mean']); sA = np.array(V[k + '_s0']['std'])
    M = np.array(V[k]['mean']); sM = np.array(V[k]['std']); nb = n_all - n_s
    mB = (n_all * M - n_s * A) / nb
    SS = n_all * (sM ** 2 + M ** 2) - n_s * (sA ** 2 + A ** 2)
    var = np.clip(SS / nb - mB ** 2, 0, None)
    return {'mean': mB.tolist(), 'std': np.sqrt(var).tolist(), 'auc': V[k]['auc'][n_s:]}
V['ns_base8'] = _no_score('sub_base8', 6, 3)
V['ns_msd0'] = _no_score('sub_msd0', 5, 2)

INK = '#1b1f2a'; MUTED = '#5b6270'; GRID = '#dcdfe3'
BLUE = '#2a78d6'; ORANGE = '#eb6834'; AQUA = '#1baf7a'; GRAY = '#8a919b'
rcParams.update({'font.family': 'serif', 'font.serif': ['cmr10', 'DejaVu Serif'], 'mathtext.fontset': 'cm', 'axes.formatter.use_mathtext': True,
                 'axes.unicode_minus': False, 'font.size': 12, 'axes.edgecolor': '#222222', 'axes.linewidth': 0.9, 'axes.labelcolor': '#111111',
                 'xtick.color': '#222222', 'ytick.color': '#222222', 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
                 'text.color': '#111111', 'axes.facecolor': 'white', 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
                 'axes.grid': True, 'grid.color': '#c8c8c8', 'grid.linewidth': 0.5, 'grid.linestyle': ':', 'axes.axisbelow': True,
                 'legend.frameon': True, 'legend.framealpha': 0.9, 'legend.edgecolor': '#bbbbbb', 'legend.fancybox': False, 'pdf.fonttype': 42})
G = np.linspace(0, 1, 101)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches='tight'); plt.close(fig)

def runs(prefix):
    return [P1[k] for k in sorted(P1) if k.startswith(prefix + '_s')]

def ms(arrs):
    a = np.array(arrs, float); return a.mean(0), a.std(0, ddof=1)

def band(ax, m, s, c, label, ls='-'):
    m = np.array(m); ax.plot(G, m, ls, color=c, lw=2.2, label=label)
    if s is not None:
        s = np.array(s); ax.fill_between(G, m - s, m + s, color=c, alpha=0.35, lw=0)

def roc_axes(ax, title):
    ax.plot([0, 1], [0, 1], ':', color=MUTED, lw=1.4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal', adjustable='box'); ax.set_title(title, fontsize=12)

# ---- steps 1 and 2: first-version taggers, training curves and ROC ------------------
def tagger_fig(prefix, color, name, out, ylim):
    R = runs(prefix); fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6))
    ep = np.arange(1, 71)
    for i, r in enumerate(R):
        a.plot(ep, r['curves']['val_auc'], '-', color=color, lw=1.1, alpha=0.45, label='validation, each seed' if i == 0 else None)
    m, _ = ms([r['curves']['val_auc'] for r in R]); a.plot(ep, m, '-', color=color, lw=2.4, label='validation, mean')
    mt, _ = ms([r['curves']['train_auc'] for r in R]); a.plot(ep, mt, '--', color=color, lw=2.0, label='training, mean')
    a.set_xlabel('epoch'); a.set_ylabel('AUC'); a.set_ylim(*ylim); a.set_xlim(1, 70); a.legend(loc='lower right', fontsize=11)
    a.set_title('Training curves (3 seeds)', loc='left', fontsize=14)
    m, s = ms([r['roc'] for r in R]); band(b, m, s, color, None)
    au = np.array([r['test_auc'] for r in R]); roc_axes(b, 'Test ROC (%s)' % name)
    b.set_xlabel('false positive rate'); b.set_ylabel('true positive rate')
    b.text(0.97, 0.06, 'AUC %.3f $\\pm$ %.3f' % (au.mean(), au.std(ddof=1)), ha='right', fontsize=14)
    fig.tight_layout(w_pad=2.5); save(fig, out)

tagger_fig('jetimg', BLUE, 'Z vs gluon, jetImage', 'ch3_step1.pdf', (0.45, 0.95))
tagger_fig('zjbbbb', ORANGE, 'Z+jet vs bbbb', 'ch3_step2.pdf', (0.5, 0.72))

# ---- first taggers on ttZ / ttbb jets ------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(11, 3.7), gridspec_kw={'width_ratios': [1, 1, 1.05]})
bins = np.linspace(0, 1, 41); bw = bins[1] - bins[0]
for ax, key, title in ((axs[0], 'jetimg_s1', 'jetImage tagger (seed 1)'), (axs[1], 'zjbbbb_s1', 'Z+jet vs bbbb tagger (seed 1)')):
    h = P1[key]['score_hist']
    for sk, col, lab in (('score_sig', AQUA, 'ttZ (Z jets)'), ('score_bkg', MUTED, 'ttbb (bb jets)')):
        d = np.array(h[sk], float); d = d / d.sum() / bw; ax.stairs(d, bins, color=col, lw=2.0, label=lab)
    ax.set_xlabel('tagger output p'); ax.set_xlim(0, 1); ax.set_title(title, loc='left', fontsize=12.5); ax.legend(loc='upper right', fontsize=10.5)
axs[1].set_ylim(0, 5.3); axs[0].set_ylabel('density')
ax = axs[2]
for prefix, col, lab in (('jetimg', BLUE, 'jetImage'), ('zjbbbb', ORANGE, 'Z+jet/bbbb')):
    R = runs(prefix); m, s = ms([r['ttz_ttbb_roc'] for r in R]); au = np.array([r['ttz_ttbb_auc'] for r in R])
    band(ax, m, s, col, '%s  %.3f $\\pm$ %.3f' % (lab, au.mean(), au.std(ddof=1)))
roc_axes(ax, 'ttZ vs ttbb jets, 3 seeds'); ax.set_xlabel('ttbb efficiency (false positive rate)'); ax.set_ylabel('ttZ efficiency')
ax.legend(loc='lower right', fontsize=10.5, title='AUC, ttZ vs ttbb', title_fontsize=10.5)
fig.tight_layout(w_pad=2); save(fig, 'ch3_ttz_ttbb.pdf')

# ---- ablation of the input (BDT) -----------------------------------------------------
fig, ax = plt.subplots(figsize=(7.4, 3.9))
labels = ['neighbour differences\n(first-version input)', 'absolute features of\ntop-10 particles', '5 jet-level features\nonly', 'all three\ncombined']
vals = [0.678, 0.780, 0.864, 0.865]; x = np.arange(4)
ax.bar(x, vals, width=0.62, color=[MUTED, BLUE, BLUE, BLUE])
for xi, v in zip(x, vals): ax.text(xi, v + 0.012, '%.3f' % v, ha='center', fontsize=13)
ax.axhline(0.5, color=MUTED, lw=1.2, ls=':'); ax.text(3.45, 0.505, 'chance', ha='right', va='bottom', fontsize=11, color=MUTED)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11); ax.set_ylim(0.4, 0.95); ax.set_ylabel('BDT test AUC, Z+jet vs bbbb'); ax.grid(axis='x', visible=False)
fig.tight_layout(); save(fig, 'ch3_ablation.pdf')

# ---- validation inversion on the new input ---------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, key, t in zip(axs, ('v2_s1', 'v2_s3'), ('seed 1', 'seed 3')):
    c = P1[key]['curves']; ep = np.arange(1, 71)
    ax.plot(ep, c['val_auc'], color=BLUE, lw=2.2, label='validation (dropout off)')
    ax.plot(ep, c['train_auc'], '--', color=MUTED, lw=2.0, label='training (dropout on)')
    ax.axhline(0.5, color=MUTED, ls=':', lw=1.2); ax.set_ylim(0.2, 0.9); ax.set_xlim(1, 70); ax.set_xlabel('epoch'); ax.set_title(t, loc='left', fontsize=14)
axs[0].set_ylabel('AUC'); axs[0].legend(loc='center right', fontsize=11)
fig.tight_layout(w_pad=2.5); save(fig, 'ch3_v2.pdf')

# ---- QGNN+ jet ROC ---------------------------------------------------------------------
fig, a = plt.subplots(figsize=(5.2, 5.0))
for k, l, c in (('v3lr', 'QGNN+, 2 layers', ORANGE), ('L1r', 'QGNN+, 1 layer', BLUE)):
    v = P2['jet'][k]; band(a, v['mean'], v['std'], c, '%s  %.3f $\\pm$ %.3f' % (l, np.mean(v['auc']), np.std(v['auc'], ddof=1)))
roc_axes(a, 'Jet-level ROC, 181-number input'); a.set_xlabel('false positive rate (bbbb)'); a.set_ylabel('true positive rate (Z+jet)'); a.legend(loc='lower right', fontsize=11)
fig.tight_layout(); save(fig, 'ch3_jet_roc.pdf')

# ---- comparison with ParT and ParticleNet -----------------------------------------------
d = [('QGNN+\n(2 layers)', 0.8665, 0.0007, ORANGE), ('QGNN+\n(1 layer)', 0.8688, 0.0010, ORANGE), ('classical GNN\n(matched)', 0.8669, 0.0009, GRAY),
     ('ParT,\nQGNN input', 0.8398, 0.0045, '#c9ccd2'), ('ParticleNet,\nQGNN input', 0.8491, 0.0030, '#c9ccd2'),
     ('ParT,\nraw particles', 0.8831, 0.0012, BLUE), ('ParticleNet,\nraw particles', 0.8860, 0.0004, AQUA)]
fig, a = plt.subplots(figsize=(11, 3.9))
for i, (l, m, s, c) in enumerate(d):
    a.bar(i, m, 0.6, color=c); a.errorbar(i, m, yerr=s, color=INK, capsize=4, lw=1.3); a.text(i, m + s + 0.0012, '%.3f $\\pm$ %.4f' % (m, s), ha='center', fontsize=11.5)
a.set_xticks(range(len(d))); a.set_xticklabels([x[0] for x in d], fontsize=11); a.set_ylim(0.82, 0.9); a.set_xlim(-0.6, len(d) - 0.4)
a.set_ylabel('test AUC, Z+jet vs bbbb'); a.grid(axis='x', visible=False)
fig.tight_layout(); save(fig, 'ch3_cmp.pdf')

# ---- data efficiency ---------------------------------------------------------------------
fig, a = plt.subplots(figsize=(7.4, 4.0)); N = [100, 300, 1000, 3000, 9996]
for k, l, c, mk in (('qgnn', 'QGNN+', ORANGE, 'o'), ('classical_default', 'classical GNN, default', GRAY, 's'), ('classical_matched', 'classical GNN, matched hyperparameters', BLUE, '^')):
    dd = P2['dataeff'][k]; xs = [n for n in N if str(n) in dd]; m = [np.mean(dd[str(n)]) for n in xs]
    s = [np.std(dd[str(n)], ddof=1) if len(dd[str(n)]) > 1 else 0 for n in xs]
    a.errorbar(xs, m, yerr=s, color=c, marker=mk, lw=2.0, ms=7, capsize=3.5, label=l)
a.set_xscale('log'); a.set_xlabel('training jets'); a.set_ylabel('test AUC'); a.legend(loc='lower right', fontsize=11)
fig.tight_layout(); save(fig, 'ch3_dataeff.pdf')

# ---- event ROC (VQC and BDT) -------------------------------------------------------------
fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 5.0))
cfg = [('ns_base8', 'jet features', GRAY), ('ns_msd0', '+ msd0', BLUE), ('sub_base8_s0', '+ tagger score q', ORANGE), ('sub_msd0_s0', '+ msd0 + q', AQUA)]
for k, l, c in cfg:
    v = V[k]; band(a, v['mean'], v.get('std'), c, '%s  %.3f' % (l, np.mean(v['auc'])))
bk = [('base8', 'jet features', GRAY), ('base8+msd0', '+ msd0', BLUE), ('base8+q', '+ tagger score q', ORANGE), ('base8+msd0+q', '+ msd0 + q', AQUA)]
for k, l, c in bk:
    v = P2['event_bdt'][k]; r = BK[k]
    b.plot(G2, RB[r + '|mean'], color=c, lw=2.2, label='%s  %.3f' % (l, np.mean(v['auc']))); b.fill_between(G2, RB[r + '|lo'], RB[r + '|hi'], color=c, alpha=0.35, lw=0)
for x, t in ((a, 'Event VQC (ttZ vs ttbb), test'), (b, 'Event BDT, same features, test (band: bootstrap 68%)')):
    roc_axes(x, t); x.set_xlabel('false positive rate (ttbb)'); x.set_ylabel('true positive rate (ttZ)'); x.legend(loc='lower right', fontsize=11)
fig.tight_layout(w_pad=2.5); save(fig, 'ch3_event_roc.pdf')

# ---- background rejection (1 / false positive rate) at fixed signal efficiency ---------------
def fpr_at(m, t):
    m = np.array(m); i = np.searchsorted(m, t)
    return G[i - 1] + (t - m[i - 1]) / (m[i] - m[i - 1]) * (G[i] - G[i - 1])

def cell(m, s, t):
    m = np.array(m)
    if s is None:
        return '$%.1f$' % (1 / fpr_at(m, t))
    s = np.array(s)
    r = 1 / fpr_at(m, t); hi = 1 / fpr_at(np.clip(m + s, 0, 1), t); lo = 1 / fpr_at(np.clip(m - s, 0, 1), t)
    return '$%.1f^{+%.1f}_{-%.1f}$' % (r, max(hi - r, 0), max(r - lo, 0))

EFF = (0.6, 0.8, 0.9)
rows = [('\\lr{QGNN+}، دو لایه', P2['jet']['v3lr']['mean'], P2['jet']['v3lr']['std']),
        ('\\lr{QGNN+}، یک لایه', P2['jet']['L1r']['mean'], P2['jet']['L1r']['std'])]
ev = [('جداکننده \\lr{VQC}، ویژگی‌های جت', V['ns_base8']), ('\\lr{VQC}، با امتیاز', V['sub_base8_s0']), ('\\lr{VQC}، با \\lr{msd0}', V['ns_msd0']), ('\\lr{VQC}، با \\lr{msd0} و امتیاز', V['sub_msd0_s0'])]
evb = [('جداکننده \\lr{BDT}، ویژگی‌های جت', 'base8'), ('\\lr{BDT}، با امتیاز', 'base8+q'), ('\\lr{BDT}، با \\lr{msd0}', 'base8+msd0'), ('\\lr{BDT}، با \\lr{msd0} و امتیاز', 'base8+msd0+q')]
def block(rs):
    return ''.join('%s & %s \\\\\n' % (n, ' & '.join(cell(m, s, t) for t in EFF)) for n, m, s in rs)
txt = ('\\begin{tabular}{lccc}\n\\toprule\nمدل & \\lr{60\\%} & \\lr{80\\%} & \\lr{90\\%} \\\\\n\\midrule\n'
       + block(rows) + '\\midrule\n' + block([(n, v['mean'], v.get('std')) for n, v in ev]))
def rcell2(k, t):
    r = BK[k]; m = RB[r + '|mean']; lo = RB[r + '|lo']; hi = RB[r + '|hi']; g = RB['G']
    def rej(c):
        i = np.searchsorted(c, t); f = g[i - 1] + (t - c[i - 1]) / (c[i] - c[i - 1]) * (g[i] - g[i - 1]); return 1 / f
    x = rej(m); return '$%.1f^{+%.1f}_{-%.1f}$' % (x, max(rej(hi) - x, 0), max(x - rej(lo), 0))
bdt = None
txt += ''.join('%s & %s \\\\\n' % (n, ' & '.join(rcell2(k, t) for t in EFF)) for n, k in evb) + '\\bottomrule\n\\end{tabular}\n'
open(os.path.join(HERE, '..', 'report', 'ch3_rej.tex'), 'w', encoding='utf8').write(txt)
print('done')

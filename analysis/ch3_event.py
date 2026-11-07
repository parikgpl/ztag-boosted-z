"""Event-level models of chapter 3 (QGNN on 10 objects, classical controls): figures and tables.
Input : analysis/ch3_data/event/{evq_summary.csv, out_evq_main/*.npz, out_evobj_main/*.npz}
        (per-event test predictions of the cluster runs; 6012 test events whose first jet has a tagger score)
Output: report/figs/ch3_ev_*.pdf, report/ch3_ev_*.tex
Run   : python3 analysis/ch3_event.py   (from the repository root)
"""
import os, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import roc_auc_score as auc, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, 'ch3_data', 'event')
OUT = os.path.join(HERE, '..', 'report'); FIG = os.path.join(OUT, 'figs')
S = pd.read_csv(os.path.join(D, 'evq_summary.csv'))
INK = '#1b1f2a'; MUTED = '#5b6270'; GRID = '#dcdfe3'; BLUE = '#2a78d6'; ORANGE = '#eb6834'; AQUA = '#1baf7a'; GRAY = '#8a919b'
rcParams.update({'font.family': 'serif', 'font.serif': ['cmr10', 'DejaVu Serif'], 'mathtext.fontset': 'cm', 'axes.formatter.use_mathtext': True,
                 'axes.unicode_minus': False, 'font.size': 12, 'axes.edgecolor': '#222222', 'axes.linewidth': 0.9, 'axes.labelcolor': '#111111',
                 'xtick.color': '#222222', 'ytick.color': '#222222', 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
                 'text.color': '#111111', 'axes.facecolor': 'white', 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
                 'axes.grid': True, 'grid.color': '#c8c8c8', 'grid.linewidth': 0.5, 'grid.linestyle': ':', 'axes.axisbelow': True,
                 'legend.frameon': True, 'legend.framealpha': 0.9, 'legend.edgecolor': '#bbbbbb', 'legend.fancybox': False, 'pdf.fonttype': 42})
G = np.linspace(0, 1, 201); NB = 400
rng = np.random.default_rng(0)

def load(folder, name, seeds=(1, 2, 3)):
    ps = []
    for s in seeds:
        z = np.load(os.path.join(D, folder, name % s)); ps.append(z['p']); y = z['y']
    return y, np.mean(ps, 0)

def roc_on_grid(y, p):
    f, t, _ = roc_curve(y, p); return np.interp(G, f, t)

def boot(y, p):
    """ensemble ROC with bootstrap 95% band (resampling test events) and rejection at fixed signal efficiency"""
    n = len(y); curves = []; auc_b = []
    for _ in range(NB):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max(): continue
        curves.append(roc_on_grid(y[i], p[i])); auc_b.append(auc(y[i], p[i]))
    c = np.array(curves)
    return dict(roc=roc_on_grid(y, p), lo=np.percentile(c, 2.5, 0), hi=np.percentile(c, 97.5, 0), auc=auc(y, p), auc_ci=np.percentile(auc_b, [2.5, 97.5]))

def rejection(y, p, eff):
    f, t, _ = roc_curve(y, p); fpr = np.interp(eff, t, f); return 1 / fpr

def rej_ci(y, p, effs):
    n = len(y); r = []
    for _ in range(NB):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max(): continue
        r.append(rejection(y[i], p[i], effs))
    r = np.array(r); return np.percentile(r, [2.5, 97.5], 0)

def lr_of(model, variant, layers, tag, fmap='rot', lat=4):
    q = S[(S.model == model) & (S.variant == variant) & (S.layers == layers) & (S.fmap == fmap) & (S.lat == lat) & (S.tag == tag)]
    return q.iloc[0]

def lrs(v): return ('%g' % v)

TAGN = [('none', 'بدون امتیاز شناساگر'), ('qgnnp', 'امتیاز \\lr{QGNN+}'), ('cgnn', 'امتیاز شبکه گرافی کلاسیک'),
        ('partq', 'امتیاز \\lr{ParT}، ورودی \\lr{QGNN+}'), ('pnq', 'امتیاز \\lr{ParticleNet}، ورودی \\lr{QGNN+}'),
        ('part', 'امتیاز \\lr{ParT}، ذره‌های خام'), ('pn', 'امتیاز \\lr{ParticleNet}، ذره‌های خام')]

def ci(r): return '$%.3f\\,[%.3f,\\,%.3f]$' % (r.ens_auc, r.lo, r.hi)
def gain(r):
    return '---' if pd.isna(r.delta_vs_none) else '$%+.3f\\,[%+.3f,\\,%+.3f]$' % (r.delta_vs_none, r.d_lo, r.d_hi)

# ---- table 1: QGNN on events and its classical-encoder control, by input score ---------------------------------
rows = ''
for t, nm in TAGN:
    a = lr_of('qgnn', 'knn', 2, t); b = lr_of('cgnn_enc', 'knn', 2, t)
    rows += '%s & %s & %s & %s \\\\\n' % (nm, ci(a), gain(a), ci(b)) if False else '%s & %s & %s & %s & %s \\\\\n' % (nm, ci(a), gain(a), ci(b), gain(b))
open(os.path.join(OUT, 'ch3_ev_tab1.tex'), 'w', encoding='utf8').write(
    '\\begin{tabular}{lcccc}\n\\toprule\n & \\multicolumn{2}{c}{\\lr{QGNN} رویدادی} & \\multicolumn{2}{c}{کنترل کلاسیک (بدون مدار)} \\\\\n'
    '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nامتیاز ورودی & \\lr{AUC} & افزایش نسبت به بدون امتیاز & \\lr{AUC} & افزایش \\\\\n\\midrule\n' + rows + '\\bottomrule\n\\end{tabular}\n')

# ---- table 2: ablations of the event QGNN (without score / with QGNN+ score) -----------------------------------
abl = [('پایه: دو لایه، گراف نزدیک‌ترین همسایه، ۴ عدد نهان', ('knn', 2, 'rot', 4)),
       ('یک لایه', ('knn', 1, 'rot', 4)), ('سه لایه', ('knn', 3, 'rot', 4)),
       ('گراف کامل (بدون انتخاب همسایه)', ('dense', 2, 'rot', 4)), ('بدون عملگرهای \\lr{CRZ}', ('nocrz', 2, 'rot', 4)),
       ('بدون تحول ایزینگ', ('noising', 2, 'rot', 4)),
       ('۱ عدد نهان برای هر کیوبیت', ('knn', 2, 'rot', 1)), ('۲ عدد نهان', ('knn', 2, 'rot', 2)), ('۸ عدد نهان', ('knn', 2, 'rot', 8)),
       ('بارگذاری داده فقط یک‌بار (بدون بارگذاری مجدد)', ('knn', 2, 'once', 4))]
rows = ''
for nm, (v, L, fm, lat) in abl:
    a = lr_of('qgnn', v, L, 'none', fm, lat); b = lr_of('qgnn', v, L, 'qgnnp', fm, lat); rows += '%s & %s & %s \\\\\n' % (nm, ci(a), ci(b))
open(os.path.join(OUT, 'ch3_ev_tab2.tex'), 'w', encoding='utf8').write(
    '\\begin{tabular}{lcc}\n\\toprule\nتغییر نسبت به مدل پایه & بدون امتیاز & با امتیاز \\lr{QGNN+} \\\\\n\\midrule\n' + rows + '\\bottomrule\n\\end{tabular}\n')

# ---- classical event models on the same objects (per-event predictions) ---------------------------------------
E = pd.read_csv(os.path.join(D, 'evobj_main_summary.csv'))   # classical models on the same events (agg_evobj.py, 1000 bootstrap resamples)
MOD = [('bdt', '\\lr{BDT} (۴۱ ویژگی)'), ('mlp', 'شبکه چندلایه (\\lr{MLP})'), ('gnn', 'شبکه گرافی'), ('pnet', 'شبیه \\lr{ParticleNet}'), ('trf', 'ترنسفورمر')]
def ecell(m, t):
    r = E[(E.model == m) & (E.tag == t)].iloc[0]; return '$%.3f\\,[%.3f,\\,%.3f]$' % (r.auc, r.lo, r.hi)
rows = ''
for m, nm in MOD:
    rows += '%s & %s \\\\\n' % (nm, ' & '.join(ecell(m, t) for t in ('none', 'qgnnp', 'part', 'pn')))
    if m == 'bdt': rows += '\\midrule\n'
a = lr_of('qgnn', 'knn', 2, 'none'); b = lr_of('qgnn', 'knn', 2, 'qgnnp'); c = lr_of('qgnn', 'knn', 2, 'part'); d = lr_of('qgnn', 'knn', 2, 'pn')
rows += '\\midrule\n\\lr{QGNN} رویدادی (مدار کوانتومی) & %s & %s & %s & %s \\\\\n' % (ci(a), ci(b), ci(c), ci(d))
open(os.path.join(OUT, 'ch3_ev_tab3.tex'), 'w', encoding='utf8').write(
    '\\begin{tabular}{lcccc}\n\\toprule\nمدل رویدادی & بدون امتیاز & امتیاز \\lr{QGNN+} & \\lr{ParT}، ذره‌های خام & \\lr{ParticleNet}، ذره‌های خام \\\\\n\\midrule\n' + rows + '\\bottomrule\n\\end{tabular}\n')

# ---- ROC with bootstrap band + rejection ------------------------------------------------------------------------
cfg = [('none', 'no jet score', GRAY), ('qgnnp', '+ QGNN+ score', ORANGE), ('pn', '+ ParticleNet score (raw particles)', AQUA), ('part', '+ ParT score (raw particles)', BLUE)]
fig, axs = plt.subplots(1, 2, figsize=(10.4, 5.0)); EFF = np.array([0.6, 0.8, 0.9]); rej_rows = ''
RJ = {}
for ax, (folder, pat, ttl) in zip(axs, (('out_evq_main', 'qgnn_knn_L2_%s_s%%d_lr%s.npz', 'Event QGNN (10 qubits)'), ('out_evobj_main', 'gnn_%s_s%%d_lr0.001.npz', 'Classical GNN on the same 10 objects'))):
    for t, lab, col in cfg:
        if folder == 'out_evq_main':
            lr = lrs(lr_of('qgnn', 'knn', 2, t).lr); y, p = load(folder, pat % (t, lr))
        else:
            y, p = load(folder, pat % t)
        b = boot(y, p); ax.plot(G, b['roc'], color=col, lw=2.2, label='%s  %.3f' % (lab, b['auc'])); ax.fill_between(G, b['lo'], b['hi'], color=col, alpha=0.35, lw=0)
        RJ[(folder, t)] = (y, p)
    ax.plot([0, 1], [0, 1], ':', color=MUTED, lw=1.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal', adjustable='box'); ax.set_title(ttl, fontsize=12)
    ax.set_xlabel('false positive rate (ttbb)'); ax.set_ylabel('true positive rate (ttZ)'); ax.legend(loc='lower right', fontsize=9.5)
fig.tight_layout(w_pad=2.5); fig.savefig(os.path.join(FIG, 'ch3_ev_roc.pdf'), bbox_inches='tight'); plt.close(fig)

def rcell(y, p):
    r = rejection(y, p, EFF); lo, hi = rej_ci(y, p, EFF)
    return ' & '.join('$%.1f\\,[%.1f,\\,%.1f]$' % (r[k], lo[k], hi[k]) for k in range(3))
rr = ''
for folder, nm in (('out_evq_main', '\\lr{QGNN} رویدادی'), ('out_evobj_main', 'شبکه گرافی کلاسیک')):
    for t, tn in (('none', 'بدون امتیاز'), ('qgnnp', 'امتیاز \\lr{QGNN+}'), ('part', 'امتیاز \\lr{ParT}')):
        if (folder, t) not in RJ:
            lr = lrs(lr_of('qgnn', 'knn', 2, t).lr) if folder == 'out_evq_main' else '0.001'
            RJ[(folder, t)] = load(folder, ('qgnn_knn_L2_%s_s%%d_lr%s.npz' % (t, lr)) if folder == 'out_evq_main' else ('gnn_%s_s%%d_lr0.001.npz' % t))
        y, p = RJ[(folder, t)]; rr += '%s، %s & %s \\\\\n' % (nm, tn, rcell(y, p))
    rr += '\\midrule\n' if folder == 'out_evq_main' else ''
open(os.path.join(OUT, 'ch3_ev_rej.tex'), 'w', encoding='utf8').write(
    '\\begin{tabular}{lccc}\n\\toprule\nمدل & \\lr{60\\%} & \\lr{80\\%} & \\lr{90\\%} \\\\\n\\midrule\n' + rr + '\\bottomrule\n\\end{tabular}\n')

# ---- bar chart: effect of the input score for QGNN vs classical-encoder control -------------------------------------
lab = ['no score', 'QGNN+', 'classical\nGNN', 'ParT\n(QGNN inp.)', 'PNet\n(QGNN inp.)', 'ParT\n(raw)', 'PNet\n(raw)']
fig, a = plt.subplots(figsize=(11, 4)); w = 0.38
for k, (mod, col, nm) in enumerate((('qgnn', ORANGE, 'event QGNN'), ('cgnn_enc', BLUE, 'classical control (same encoder, no circuit)'))):
    for i, (t, _) in enumerate(TAGN):
        r = lr_of(mod, 'knn', 2, t); x = i + (k - 0.5) * w
        a.bar(x, r.ens_auc, w * 0.95, color=col, label=nm if i == 0 else None); a.errorbar(x, r.ens_auc, yerr=[[r.ens_auc - r.lo], [r.hi - r.ens_auc]], color=INK, lw=1.2, capsize=3)
a.set_xticks(range(7)); a.set_xticklabels(lab, fontsize=10.5); a.set_ylim(0.7, 0.9); a.set_ylabel('test AUC, ttZ vs ttbb'); a.grid(axis='x', visible=False); a.legend(loc='upper left', fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'ch3_ev_tags.pdf'), bbox_inches='tight'); plt.close(fig)
print(open(os.path.join(OUT, 'ch3_ev_tab1.tex'), encoding='utf8').read()); print(open(os.path.join(OUT, 'ch3_ev_tab3.tex'), encoding='utf8').read()); print(open(os.path.join(OUT, 'ch3_ev_rej.tex'), encoding='utf8').read())

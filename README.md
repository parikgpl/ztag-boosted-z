# ztag-boosted-z — دادهٔ برچسب‌زنی Z→bb̄ پرانرژی

این ریپو همهٔ مراحل ساخت یک دادهٔ شبیه‌سازی‌شده برای شناساگر Z→bb̄ پرانرژی (boosted)، اسکریپت‌های تحلیل و شکل‌ها، و گزارش فارسی (XePersian LaTeX) را نگه می‌دارد. داده‌های بزرگ (فایل‌های LHE، HepMC، ROOT و h5) در ریپو نیست و روی EOS (`/eos/user/p/pkangazi/ZTAG_project`) می‌ماند.

## زنجیرهٔ شبیه‌سازی

MadGraph 3.5.3 ← MadSpin ← Pythia 8.312 ← Delphes (کارت CMS) ← FastJet (anti-kT با R=0.8)، در 13 TeV.

چهار نمونه داریم و برای هر کدام ۲۰ job با ۱۰۰۰ رویداد (۲۰۰۰۰ رویداد):

- `zj` (Z+jet) و `ttZ`: سیگنال، با برش `pt_min_pdg {23:400}` روی Z و واپاشی Z→bb̄ با MadSpin.
- `ttbb` و `bbbb`: پس‌زمینه. در ttbb یک زوج bb̄ با pT>400 GeV و ΔR(b,b̄)<0.8 لازم است. در bbbb زوج‌بندی با کمترین مجموع ΔR انجام می‌شود و هر دو زوج باید pT>400 و ΔR<0.8 داشته باشند (تابع `bbbb_cut` در `cuts.f`). برای bbbb کارت MadSpin نداریم.

بذر هر job برابر `ID*10000+J+1000` است و در MadGraph (`iseed`)، MadSpin، Pythia و Delphes یکسان به کار می‌رود و در `seeds.txt` هر job ثبت می‌شود. خروجی هر مرحله نگه داشته می‌شود (`stages/v5/`) تا Z در مرحله‌های مختلف قابل مقایسه باشد.

## ساختار ریپو

- `pipeline/madgraph/`: کارت‌های ساخت فرایندها (`make_*.mg5`)، کارت‌های MadSpin (`madspin_*.dat`) و پوشهٔ هر فرایند `MG_*/` با `proc_card_mg5.dat` و `run_card.dat` (برای ttbb و bbbb با `cuts.f`).
- `pipeline/pythia/shower_lhe.cc`: بارش پارتونی با Pythia از فایل LHE و خروجی HepMC3.
- `pipeline/delphes/delphes_card_CMS.tcl`: کارت شبیه‌سازی آشکارساز CMS.
- `pipeline/fastjet/jetfeat4.cc`: همهٔ جت‌های anti-kT با R=0.8 هر رویداد، برچسب‌های درستی (truth) و ویژگی‌های زیرساختار، و ۱۷ ویژگی برای ۱۰۰ ذرهٔ سخت‌تر هر جت.
- `pipeline/condor/`: `run_job5.sh` (یک job کامل)، `ztag5.sub` (فایل HTCondor) و `jobs_v5.txt` (فهرست jobها).
- `pipeline/build.sh`: کامپایل `shower_lhe` و `jetfeat4` (آزموده‌شده روی lxplus با LCG_106).
- `analysis/`: اسکریپت‌های تحلیل و شکل (پایین).
- `report/`: متن گزارش (`Report.tex`، `Report.bib`، `part*.tex`، `appendixA.tex`)، لوگوها، `figs/` (شکل‌های برداری PDF) و `Report.pdf`.

## اسکریپت‌های analysis

- `make_h5.py`: ساخت دادهٔ h5 (`jet_features.h5` و `events.h5`) از خروجی ROOT.
- `cutflow_origin.py`: جدول برش‌ها و منشأ جت‌ها (Z کامل، Z جزئی، جفت bb و ...) و شکل‌های `cutflow_stacked` و `origin_fractions`.
- `final_plots.py` و `final_plots_partial.py`: شکل‌های مجموعه‌های ۱ تا ۴ برای جت‌های کامل و جزئی.
- `plot_cons.py`: شکل ویژگی‌های ۱۰ ذرهٔ اول جت (هر ویژگی در یک صفحه).
- `stage_overview.py`: نمای کلی مقایسهٔ مرحله‌ها.
- `lhe_stages.py`: توزیع ویژگی‌های Z در خروجی MadGraph و MadSpin (شکل‌های `lhe_*`).
- `pol.py`: قطبش Z از cosθ* در دستگاه سکون Z، با f_L = 2 − 5⟨cos²θ*⟩ (فرض دو حالت T و L، خطای آماری).
- `zsplit.py`: تقسیم انرژی بین دو b، یعنی z و √(z(1−z)).
- `shat_y_zpt.py`: √ŝ، تندی y سیستم و pT زد.
- `anc.py`: بررسی اجداد ذرات (آلودگی از واپاشی تاپ).
- `jetdata.py` و `jet_numbers.py`: بارگذاری جدول جت‌ها و عددهای خلاصه (توضیح در سرفایل هر کدام).

## اجرا

1. کامپایل: `bash pipeline/build.sh`
2. ارسال jobها: `condor_submit pipeline/condor/ztag5.sub` (فهرست jobها در `jobs_v5.txt`؛ شمارهٔ jobهای bbbb پیوسته نیست: ۳۰۰–۳۰۶، ۳۰۸–۳۱۰، ۳۱۲، ۳۱۳، ۳۱۶–۳۲۰، ۳۲۴–۳۲۶ که ۳۲۶ با ۳۴۰ جایگزین شده).
3. ساخت h5: `python3 analysis/make_h5.py <ورودی fastjet/out/v5> <خروجی h5/v5>`
4. شکل‌ها و جدول‌ها: `python3 analysis/cutflow_origin.py <ورودی> <خروجی>` و بقیهٔ اسکریپت‌ها.
5. گزارش: داخل `report/` دوبار `xelatex -interaction=nonstopmode Report.tex`.

برای اجرای jetfeat4 باید `LD_LIBRARY_PATH` شامل `lib` همان view باشد.

## کدگذاری داده

نمونه: 0=zj، 1=ttZ، 2=bbbb، 3=ttbb. منشأ جت: 1=کامل، 2=جزئی. برچسب: 1=سیگنال، 0=پس‌زمینه.

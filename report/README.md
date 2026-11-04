# گزارش پیشرفت طرح: یادگیری ماشین کوانتومی مضاعف برای کشف برهمکنش‌های نوین کوارک تاپ (مرحله دوم)

ساختار: مقدمه (`intro_main.tex`)، فصل ۱ شناساگر کوانتومی جت (`chapter1.tex`)، فصل ۲ تولید داده برای ttZ و ttbb (`introduction.tex` و `part*.tex`)، مراجع (`Report.bib`)، ضمائم (`appendix*.tex`). فایل اصلی `Report.tex` است و شکل‌ها در `figs/` هستند.

## نیازمندی‌ها
XeLaTeX با بسته‌های `xepersian` و `biblatex`، برنامه `biber`، قلم‌های B Nazanin و TeX Gyre Termes.

## اجرا
```bash
xelatex Report.tex && biber Report && xelatex Report.tex && xelatex Report.tex
```
خروجی `Report.pdf` است. دو بار آخر `xelatex` برای درست‌شدن ارجاع‌ها و فهرست‌هاست.

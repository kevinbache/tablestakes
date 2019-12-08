from pathlib import Path

from weasyprint import HTML, CSS

this_dir = Path(__file__).resolve().absolute().parent
out_dir = this_dir

input_dir = Path('/Users/kevin/projects/tablestakes/lib/samples_weasyprint/invoice/')

html_str = open(input_dir / 'invoice.html', mode='r').read()
html = HTML(string=html_str)

css_str = open(input_dir / 'invoice.css', mode='r').read()
css = CSS(string=css_str)

html.write_pdf(
	out_dir / 'test_invoice.pdf',
	stylesheets=[css],
)
print('done')
Ω

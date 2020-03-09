import weasyprint

html_str = """
<html>
…
</html>
"""

css_str = """@page {
      size: Letter; /* Change from the default size of A4 */
      margin: 2.5cm; /* Set margin on each page */
}
"""

html = weasyprint.HTML(string=html_str)

html.write_pdf(
    target='/tmp/simple_weasyprint.pdf',
    stylesheets=[weasyprint.CSS(string=css_str)],
)
# import pdfkit
# pdfkit.from_string(html, output_path='pdfkit_test.pdf')
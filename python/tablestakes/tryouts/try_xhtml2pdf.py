'''
https://gearheart.io/blog/how-generate-pdf-files-python-xhtml2pdf-weasyprint-or-unoconv/
'''
from xhtml2pdf import pisa
import cStringIO as StringIO

from django.template.loader import get_template
from django.template import Context

# ref: https://www.w3schools.com/html/html_tables.asp
html = '''
<table style="width:100%">
  <tr>
    <th>Firstname</th>
    <th>Lastname</th>
    <th>Age</th>
  </tr>
  <tr>
    <td>Jill</td>
    <td>Smith</td>
    <td>50</td>
  </tr>
  <tr>
    <td>Eve</td>
    <td>Jackson</td>
    <td>94</td>
  </tr>
</table>
'''

def html_to_pdf_directly(request):
	template = get_template("template_name.html")
	context = Context({'pagesize': 'A4'})
	html = template.render(context)
	result = StringIO.StringIO()
	pdf = pisa.pisaDocument(StringIO.StringIO(html), dest=result)
	if not pdf.err:
		return HttpResponse(result.getvalue(), content_type='application/pdf')
	else: return HttpResponse('Errors')

"""Render the repository's four-page robotics section from its Markdown source."""
from pathlib import Path
import argparse,re,html
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Image,Table,TableStyle
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4

ap=argparse.ArgumentParser();ap.add_argument('--markdown',type=Path,required=True);ap.add_argument('--figure',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib
fontroot=Path(matplotlib.get_data_path())/'fonts/ttf'
for name,file in [('Report','DejaVuSans.ttf'),('ReportBold','DejaVuSans-Bold.ttf'),('ReportItalic','DejaVuSans-Oblique.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(fontroot/file)))
pdfmetrics.registerFontFamily('Report',normal='Report',bold='ReportBold',italic='ReportItalic',boldItalic='ReportBold')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyReport',fontName='Report',fontSize=9.5,leading=13,spaceAfter=7))
styles.add(ParagraphStyle(name='HeadReport',fontName='ReportBold',fontSize=12,leading=15,spaceBefore=7,spaceAfter=8,keepWithNext=True,textColor=colors.HexColor('#173d52')))
styles.add(ParagraphStyle(name='TitleReport',fontName='ReportBold',fontSize=17,leading=20,spaceAfter=14,textColor=colors.HexColor('#173d52')))
styles.add(ParagraphStyle(name='SmallReport',fontName='Report',fontSize=8.5,leading=10.5,spaceAfter=5))
styles.add(ParagraphStyle(name='TableReport',fontName='Report',fontSize=8,leading=10))
base='https://github.com/abjain182002/multimodal-perception/blob/main/'
def inline(text):
    text=html.escape(text)
    def link(m):
        label,url=m.groups()
        if url.startswith('../../'):url=base+url[6:]
        return '<a href="'+url+'" color="#176080">'+label+'</a>'
    text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,text)
    text=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)',r'<i>\1</i>',text)
    text=re.sub(r'`([^`]+)`',r'<font size="7">\1</font>',text)
    return text
flow=[]; lines=a.markdown.read_text().splitlines();i=0
while i<len(lines):
    line=lines[i].strip();i+=1
    if not line:continue
    if line=='<!-- PAGEBREAK -->':flow.append(PageBreak());continue
    if line.startswith('# '):flow.append(Paragraph(inline(line[2:]),styles['TitleReport']));continue
    if line.startswith('## '):flow.append(Paragraph(inline(line[3:]),styles['HeadReport']));continue
    if line.startswith('!['):
        from PIL import Image as PILImage
        with PILImage.open(a.figure) as source: w,h=source.size
        im=Image(str(a.figure),width=490,height=490*h/w);flow.extend([Spacer(1,7),im,Spacer(1,7)]);continue
    if line.startswith('|'):
        rows=[line]
        while i<len(lines) and lines[i].startswith('|'):rows.append(lines[i]);i+=1
        rows=[r for r in rows if not re.fullmatch(r'[|\- :]+',r)]
        cells=[[Paragraph(inline(c.strip()),styles['TableReport']) for c in r.strip('|').split('|')] for r in rows]
        t=Table(cells,colWidths=[66,100,89,89,156],repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4edf2')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#173d52')),('LINEBELOW',(0,1),(-1,-1),.25,colors.lightgrey),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
        flow.extend([t,Spacer(1,8)]);continue
    while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![','<!--')):line+=' '+lines[i].strip();i+=1
    small=line.startswith(('[1]','[2]','[3]','[4]','**Figure','**Table'))
    flow.append(Paragraph(inline(line),styles['SmallReport' if small else 'BodyReport']))
def page(canvas,doc):
    canvas.setFont('Report',8);canvas.setFillColor(colors.HexColor('#53636c'))
    canvas.drawString(46,24,'MSc dissertation section draft | 23 October 2026')
    canvas.drawRightString(A4[0]-46,24,str(doc.page))
a.output.parent.mkdir(parents=True,exist_ok=True)
SimpleDocTemplate(str(a.output),pagesize=A4,leftMargin=46,rightMargin=46,topMargin=38,bottomMargin=40,title='Camera-LiDAR perception for mobile-robot obstacle alerts',author='Abhishek Jain').build(flow,onFirstPage=page,onLaterPages=page)

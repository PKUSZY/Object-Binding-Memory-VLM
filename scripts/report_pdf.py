"""Render the report's Markdown blocks into a paginated, searchable Chinese PDF."""
import html
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def render_pdf(md, output, asset_root, title):
    # Optional report-only dependencies, never installed in the inference environment.
    sys.path.append('/tmp/bindmem-report-deps')
    import markdown
    from PIL import Image as PILImage
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
    font='/usr/share/fonts/google-droid/DroidSansFallback.ttf'
    if 'BindMemCN' not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont('BindMemCN',font))
        pdfmetrics.registerFontFamily('BindMemCN',normal='BindMemCN',bold='BindMemCN',italic='BindMemCN',boldItalic='BindMemCN')
    W,H=A4; margin=40; width=W-2*margin
    base=dict(fontName='BindMemCN',fontSize=10,leading=16,wordWrap='CJK',spaceAfter=9,alignment=TA_LEFT)
    styles={'p':ParagraphStyle('Body',**base)}
    for tag,size,leading in [('h1',23,31),('h2',16,24),('h3',12,19),('h4',11,17)]:
        styles[tag]=ParagraphStyle(tag,**dict(base,fontSize=size,leading=leading,spaceBefore=15,spaceAfter=10,keepWithNext=True,textColor=colors.HexColor('#163F5D')))
    styles['cell']=ParagraphStyle('Cell',**dict(base,fontSize=8,leading=12,spaceAfter=0))
    styles['code']=ParagraphStyle('Code',**dict(base,fontSize=8.4,leading=12,backColor=colors.HexColor('#F0F4F7'),borderPadding=8,spaceBefore=6))
    def inline(element):
        out=html.escape(element.text or '')
        for child in element:
            inner=inline(child)
            if child.tag in ('strong','b'):out+='<b>'+inner+'</b>'
            elif child.tag in ('em','i'):out+='<i>'+inner+'</i>'
            elif child.tag=='br':out+='<br/>'
            elif child.tag=='a':
                href=child.get('href','')
                out+=('<a href="'+html.escape(href,quote=True)+'" color="#236890">'+inner+'</a>') if href.startswith(('https://','http://')) else inner
            elif child.tag=='code':out+='<font color="#325A70">'+inner+'</font>'
            elif child.tag!='img':out+=inner
            out+=html.escape(child.tail or '')
        return out
    def image_flow(src):
        path=Path(asset_root)/src
        with PILImage.open(path) as im: iw,ih=im.size
        scale=min(width/iw,480/ih)
        return Image(str(path),width=iw*scale,height=ih*scale,hAlign='CENTER')
    tree=ET.fromstring('<root>'+markdown.markdown(md,extensions=['tables','fenced_code'],output_format='xhtml')+'</root>')
    story=[]
    for element in tree:
        tag=element.tag
        if tag in ('h1','h2','h3','h4'):
            if tag=='h2':story.append(Spacer(1,4))
            story.append(Paragraph(inline(element),styles[tag]))
        elif tag=='p':
            imgs=element.findall('.//img')
            if imgs:
                for im in imgs:
                    before=Spacer(1,4);before.keepWithNext=True
                    picture=image_flow(im.get('src'));picture.keepWithNext=True
                    after=Spacer(1,8);after.keepWithNext=True
                    story.extend([before,picture,after])
            else:story.append(Paragraph(inline(element),styles['p']))
        elif tag=='pre':
            text=''.join(element.itertext())
            # Breaking code into short blocks permits page breaks for long actual prompts.
            for line in text.splitlines():
                story.append(Paragraph(html.escape(line).replace(' ','&#160;') or '&#160;',styles['code']))
        elif tag in ('ul','ol'):
            for i,li in enumerate(element.findall('li'),1):
                prefix=f'{i}. ' if tag=='ol' else '• '
                story.append(Paragraph(prefix+inline(li),styles['p']))
        elif tag=='table':
            rows=element.findall('.//tr');raw=[[inline(c) for c in r] for r in rows]
            n=len(raw[0]); assert all(len(r)==n for r in raw)
            # Give descriptive first columns more room; keep dense numeric appendices legible.
            weights=[1.7]+[1]*(n-1) if n>=4 else ([1.15,2.4,1.2] if n==3 else [1,1])
            if n==1:weights=[1]
            if n==9:weights=[.6,.6]+[1]*7
            cols=[width*w/sum(weights) for w in weights]
            table=Table([[Paragraph(c,styles['cell']) for c in row] for row in raw],colWidths=cols,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8F0F5')),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F7F9FB')]),
                ('GRID',(0,0),(-1,-1),.35,colors.HexColor('#CCD7DF')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),
                ('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
            story.extend([table,Spacer(1,12)])
        elif tag=='blockquote':story.append(Paragraph(''.join(element.itertext()),styles['p']))
        elif tag=='hr':story.append(PageBreak())
    def page(canvas,doc):
        canvas.saveState();canvas.setFont('BindMemCN',8);canvas.setFillColor(colors.HexColor('#607484'))
        canvas.drawString(margin,H-23,title)
        canvas.drawRightString(W-margin,22,f'BindMem · 2026-10-29 · {doc.page}')
        canvas.restoreState()
    doc=SimpleDocTemplate(str(output),pagesize=A4,leftMargin=margin,rightMargin=margin,topMargin=43,bottomMargin=39,
        title=title,author='BindMem project',pageCompression=1)
    doc.build(story,onFirstPage=page,onLaterPages=page)

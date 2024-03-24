
import PyPDF2 as pp
def extract(pdf):
    text=""
    with pdf as f:
        read=pp.PdfReader(f)
        pages=len(read.pages)
        for pagen in range(pages):
            page=read.pages[pagen]
            text+=page.extract_text()
    return text


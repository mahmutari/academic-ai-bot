from PyPDF2 import PdfReader

class PDFLoader:
    @staticmethod
    def get_pdf_text(pdf_path):
        text = ""
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
        return text
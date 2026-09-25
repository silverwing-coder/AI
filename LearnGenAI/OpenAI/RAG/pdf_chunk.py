import PyPDF2
from PIL import Image
import tabula

## pip install pdfminer.six
from pdfminer.high_level import extract_text
from PyPDF2 import PdfReader

def exact_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        print("Number of PDF pages:", len(reader.pages))
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text


pdf_file = 'data/sample.pdf'
output_folder = f"data/temp/"

# text = exact_text_from_pdf(pdf_file)
# print(text)

text = ''
def process_element(element, LTTextBox):
    global text
    if isinstance(element, LTTextBox):
        text += element.get_text()
    elif isinstance(element, LTImage):
        for child in element:
            process_element(child, LTTextBox)

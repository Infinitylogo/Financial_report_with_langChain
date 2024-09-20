import PyPDF2
import os

# Function to extract text from a PDF file in chunks
def extract_text_from_pdf(pdf_path, chunk_size=10):
    # Validate that the file exists
    if not os.path.exists(pdf_path):
        return {"error": "File not found"}
    
    text = ""
    
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            
            # Validate that the PDF has pages
            if num_pages == 0:
                return {"error": "PDF file contains no pages"}
            
            # Reading PDF in chunks
            for start in range(0, num_pages, chunk_size):
                end = min(start + chunk_size, num_pages)
                for page_num in range(start, end):
                    text += reader.pages[page_num].extract_text() or ""
                    
    except Exception as e:
        return {"error": f"Failed to extract text from PDF: {str(e)}"}
    
    # Validate that some text was extracted
    if not text.strip():
        return {"error": "No text extracted from PDF"}
    
    return text


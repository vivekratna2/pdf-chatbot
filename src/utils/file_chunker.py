from pypdf import PdfReader
from textsplitter import TextSplitter

class PDFChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        text = ""
        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() or "\n"
        except Exception as e:
            print(f"Error reading PDF file {pdf_path}: {e}")
        return text
    
    def chunk_text(self, text: str) -> list[str]:
        """
        Splits the text into chunks of specified size with overlap.
        """
        text_splitter = TextSplitter(max_token_size=self.chunk_size, remove_stopwords=False)
        chunks = text_splitter.split_text(text)
        return chunks
    
    def process_pdf(self, pdf_path: str) -> list[str]:
        """
        Processes the PDF file and returns a list of text chunks.
        """
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return []
        
        chunks = self.chunk_text(text)
        return chunks
        
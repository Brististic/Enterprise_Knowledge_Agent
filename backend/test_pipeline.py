import json
from app.services.parser import MultimodalDocumentParser

if __name__ == "__main__":
    parser = MultimodalDocumentParser()
    
    # Place any sample PDF (e.g., an annual report or technical paper) in data/uploads/
    sample_pdf = "data/uploads/sample.pdf"
    
    try:
        chunks = parser.parse_pdf(sample_pdf)
        print(f"\nSuccessfully extracted {len(chunks)} chunks.")
        print("Sample Chunk Output:")
        print(json.dumps(chunks[:2], indent=2))
    except FileNotFoundError:
        print(f"Please drop a sample PDF at '{sample_pdf}' to test.")
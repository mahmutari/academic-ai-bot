from src.loader import PDFLoader
from src.processor import TextProcessor

def main():
    pdf_path = "data/Python-plot.pdf" #
    
    # 1. Veriyi Yükle
    loader = PDFLoader()
    raw_text = loader.get_pdf_text(pdf_path)
    print(f"Ham metin boyutu: {len(raw_text)} karakter.")

    # 2. Metni Parçala (Day 2 Görevi)
    processor = TextProcessor()
    chunks = processor.split_text(raw_text)
    
    print(f"Metin {len(chunks)} adet parçaya bölündü.")
    print(f"İlk parça örneği:\n{chunks[0][:200]}...")

if __name__ == "__main__":
    main()
import os
from typing import Optional, List
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Load biến môi trường
load_dotenv("./.env")

class PDFVectorStore:
    """
    Class để tạo và quản lý vector store từ file PDF
    """
    
    def __init__(self, embedding_model: str = "models/embedding-001"):
        """
        Khởi tạo PDFVectorStore
        
        Args:
            embedding_model (str): Tên model embedding của Gemini
        """
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
            # print(f"✅ Embedding model ({embedding_model}) đã được khởi tạo thành công!")
        except Exception as e:
            print(f"❌ Lỗi khi khởi tạo Embedding model: {e}")
            print("💡 Vui lòng đảm bảo bạn đã đặt biến môi trường 'GOOGLE_API_KEY' và nó hợp lệ.")
            raise e
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load tài liệu từ file PDF
        
        Args:
            pdf_path (str): Đường dẫn đến file PDF
            
        Returns:
            List[Document]: Danh sách các document đã load
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"❌ Không tìm thấy file PDF: {pdf_path}")
        
        try:
            print(f"📄 Đang tải tài liệu từ PDF: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            print(f"✅ Đã tải {len(docs)} trang từ PDF.")
            return docs
        except Exception as e:
            print(f"❌ Lỗi khi tải PDF: {e}")
            raise e
    
    def split_documents(
        self, 
        docs: List[Document], 
        chunk_size: int = 1000, 
        chunk_overlap: int = 150,
        add_section_metadata: bool = True
    ) -> List[Document]:
        """
        Chia tài liệu thành các chunks nhỏ hơn
        
        Args:
            docs (List[Document]): Danh sách documents cần chia
            chunk_size (int): Kích thước mỗi chunk
            chunk_overlap (int): Số ký tự overlap giữa các chunks
            add_section_metadata (bool): Có thêm metadata section không
            
        Returns:
            List[Document]: Danh sách các chunks
        """
        try:
            print(f"✂️ Đang chia tài liệu thành chunks (size={chunk_size}, overlap={chunk_overlap})...")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            all_splits = text_splitter.split_documents(docs)
            print(f"✅ Đã chia thành {len(all_splits)} chunks.")
            
            # Thêm metadata section nếu được yêu cầu
            if add_section_metadata:
                self._add_section_metadata(all_splits)
                print("✅ Đã thêm metadata 'section' cho các chunks.")
            
            return all_splits
        except Exception as e:
            print(f"❌ Lỗi khi chia documents: {e}")
            raise e
    
    def _add_section_metadata(self, documents: List[Document]):
        """
        Thêm metadata section (beginning, middle, end) cho documents
        
        Args:
            documents (List[Document]): Danh sách documents cần thêm metadata
        """
        total_documents = len(documents)
        third = total_documents // 3
        
        for i, document in enumerate(documents):
            if i < third:
                document.metadata["section"] = "beginning"
            elif i < 2 * third:
                document.metadata["section"] = "middle"
            else:
                document.metadata["section"] = "end"
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """
        Tạo vector store từ danh sách documents
        
        Args:
            documents (List[Document]): Danh sách documents cần index
            
        Returns:
            FAISS: Vector store đã được tạo
        """
        try:
            print(f"🔍 Đang tạo vector store và index {len(documents)} chunks...")
            
            # Tạo vector store từ documents
            vector_store = FAISS.from_documents(documents, self.embeddings)
            
            print(f"✅ Đã tạo vector store và index {len(documents)} chunks thành công!")
            return vector_store
        except Exception as e:
            print(f"❌ Lỗi khi tạo vector store: {e}")
            raise e
    
    def create_vector_store_from_pdf(
        self, 
        pdf_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        add_section_metadata: bool = True
    ) -> FAISS:
        """
        Tạo vector store trực tiếp từ file PDF (hàm all-in-one)
        
        Args:
            pdf_path (str): Đường dẫn đến file PDF
            chunk_size (int): Kích thước mỗi chunk
            chunk_overlap (int): Số ký tự overlap giữa các chunks
            add_section_metadata (bool): Có thêm metadata section không
            
        Returns:
            FAISS: Vector store đã được tạo và index
        """
        try:
            # 1. Load PDF
            docs = self.load_pdf(pdf_path)
            
            # 2. Split documents
            chunks = self.split_documents(
                docs, 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                add_section_metadata=add_section_metadata
            )
            
            # 3. Create vector store
            vector_store = self.create_vector_store(chunks)
            
            print(f"🎉 Hoàn thành! Vector store đã sẵn sàng với {len(chunks)} chunks.")
            return vector_store
            
        except Exception as e:
            print(f"❌ Lỗi trong quá trình tạo vector store từ PDF: {e}")
            raise e
    
    def save_vector_store(self, vector_store: FAISS, save_path: str):
        """
        Lưu vector store vào local
        
        Args:
            vector_store (FAISS): Vector store cần lưu
            save_path (str): Đường dẫn thư mục để lưu
        """
        try:
            print(f"💾 Đang lưu vector store vào: {save_path}")
            vector_store.save_local(save_path)
            print(f"✅ Đã lưu vector store thành công!")
        except Exception as e:
            print(f"❌ Lỗi khi lưu vector store: {e}")
            raise e
    
    def load_vector_store(self, load_path: str) -> FAISS:
        """
        Load vector store từ local
        
        Args:
            load_path (str): Đường dẫn thư mục chứa vector store
            
        Returns:
            FAISS: Vector store đã được load
        """
        try:
            # print(f"📂 Đang load vector store từ: {load_path}")
            vector_store = FAISS.load_local(
                load_path, 
                self.embeddings,
                allow_dangerous_deserialization=True  # Cần thiết cho FAISS
            )
            # print(f"✅ Đã load vector store thành công!")
            return vector_store
        except Exception as e:
            print(f"❌ Lỗi khi load vector store: {e}")
            raise e


# === HÀM TIỆN ÍCH (Utility Functions) ===

def create_db_from_pdf(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embedding_model: str = "models/embedding-001",
    add_section_metadata: bool = True
) -> FAISS:
    """
    Hàm tiện ích để tạo vector store từ PDF một cách nhanh chóng
    
    Args:
        pdf_path (str): Đường dẫn đến file PDF
        chunk_size (int): Kích thước mỗi chunk
        chunk_overlap (int): Số ký tự overlap giữa các chunks
        embedding_model (str): Model embedding
        add_section_metadata (bool): Có thêm metadata section không
        
    Returns:
        FAISS: Vector store đã được tạo
    """
    pdf_vector_store = PDFVectorStore(embedding_model=embedding_model)
    return pdf_vector_store.create_vector_store_from_pdf(
        pdf_path=pdf_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_section_metadata=add_section_metadata
    )


def get_db_from_saved(
    load_path: str,
    embedding_model: str = "models/embedding-001"
) -> FAISS:
    """
    Hàm tiện ích để load vector store đã lưu
    
    Args:
        load_path (str): Đường dẫn thư mục chứa vector store
        embedding_model (str): Model embedding
        
    Returns:
        FAISS: Vector store đã được load
    """
    pdf_vector_store = PDFVectorStore(embedding_model=embedding_model)
    return pdf_vector_store.load_vector_store(load_path)


# === DEMO ===
if __name__ == "__main__":
    pdf_vs1 = PDFVectorStore()
    doc = pdf_vs1.load_pdf("data/Data_wiki_Elon_Musk.pdf")
    doc_splits = pdf_vs1.split_documents(doc)

    print(doc_splits[0].metadata)
    # vector_store = pdf_vs1.create_vector_store(doc_splits)
    # vector_store.save_local("vector_store_faiss")


    # vector_store = pdf_vs1.create_db_from_pdf("data/Data_wiki_Elon_Musk.pdf", add_section_metadata=False)
    # vector_store.save_local("/db/vector_store_faiss")

    # print("đã tạo vector store và lưu vào local tại: /db/vector_store_faiss")
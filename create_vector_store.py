import os
from typing import Optional, List
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Load environment variables
load_dotenv("./.env")

class PDFVectorStore:
    """
    Class for creating and managing vector store from PDF files
    """
    
    def __init__(self, embedding_model: str = "models/embedding-001"):
        """
        Initialize PDFVectorStore
        
        Args:
            embedding_model (str): Gemini embedding model name
        """
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
            # print(f"✅ Embedding model ({embedding_model}) initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing Embedding model: {e}")
            print("💡 Please ensure you have set the 'GOOGLE_API_KEY' environment variable and it's valid.")
            raise e
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load documents from PDF file
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            List[Document]: List of loaded documents
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"❌ PDF file not found: {pdf_path}")
        
        try:
            print(f"📄 Loading document from PDF: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            print(f"✅ Loaded {len(docs)} pages from PDF.")
            return docs
        except Exception as e:
            print(f"❌ Error loading PDF: {e}")
            raise e
    
    def split_documents(
        self, 
        docs: List[Document], 
        chunk_size: int = 1000, 
        chunk_overlap: int = 150,
        add_section_metadata: bool = True
    ) -> List[Document]:
        """
        Split documents into smaller chunks
        
        Args:
            docs (List[Document]): List of documents to split
            chunk_size (int): Size of each chunk
            chunk_overlap (int): Number of characters overlap between chunks
            add_section_metadata (bool): Whether to add section metadata
            
        Returns:
            List[Document]: List of chunks
        """
        try:
            print(f"✂️ Splitting documents into chunks (size={chunk_size}, overlap={chunk_overlap})...")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            all_splits = text_splitter.split_documents(docs)
            print(f"✅ Split into {len(all_splits)} chunks.")
            
            # Add section metadata if requested
            if add_section_metadata:
                self._add_section_metadata(all_splits)
                print("✅ Added 'section' metadata to chunks.")
            
            return all_splits
        except Exception as e:
            print(f"❌ Error splitting documents: {e}")
            raise e
    
    def _add_section_metadata(self, documents: List[Document]):
        """
        Add section metadata to documents based on page numbers
        
        This function categorizes document pages into different periods:
        - Pages 1-8: "diem_era_and_coup" (Diem era and coup period)
        - Pages 9-17: "us_escalation_and_withdrawal" (US escalation and withdrawal period)
        - Pages 18+: "war_end_and_aftermath" (War end and aftermath period)
        
        Args:
            documents (List[Document]): List of documents to add metadata to
        """
        for doc in documents:
            # PyPDFLoader automatically adds 'page' to metadata, starting from 0
            page_number = doc.metadata.get("page", 0) + 1  # Convert to page number starting from 1

            if page_number <= 8:
                doc.metadata["section"] = "diem_era_and_coup"
            elif 9 <= page_number <= 17:
                doc.metadata["section"] = "us_escalation_and_withdrawal"
            else:
                # Remaining pages from 18 -> 28
                doc.metadata["section"] = "war_end_and_aftermath"
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """
        Create vector store from list of documents
        
        Args:
            documents (List[Document]): List of documents to index
            
        Returns:
            FAISS: Created vector store
        """
        try:
            print(f"🔍 Creating vector store and indexing {len(documents)} chunks...")
            
            # Create vector store from documents
            vector_store = FAISS.from_documents(documents, self.embeddings)
            
            print(f"✅ Successfully created vector store and indexed {len(documents)} chunks!")
            return vector_store
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            raise e
    
    def create_vector_store_from_pdf(
        self, 
        pdf_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        add_section_metadata: bool = True
    ) -> FAISS:
        """
        Create vector store directly from PDF file (all-in-one function)
        
        Args:
            pdf_path (str): Path to PDF file
            chunk_size (int): Size of each chunk
            chunk_overlap (int): Number of characters overlap between chunks
            add_section_metadata (bool): Whether to add section metadata
            
        Returns:
            FAISS: Created and indexed vector store
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
            
            print(f"🎉 Complete! Vector store is ready with {len(chunks)} chunks.")
            return vector_store
            
        except Exception as e:
            print(f"❌ Error during vector store creation from PDF: {e}")
            raise e
    
    def save_vector_store(self, vector_store: FAISS, save_path: str):
        """
        Save vector store to local storage
        
        Args:
            vector_store (FAISS): Vector store to save
            save_path (str): Directory path to save to
        """
        try:
            print(f"💾 Saving vector store to: {save_path}")
            vector_store.save_local(save_path)
            print(f"✅ Vector store saved successfully!")
        except Exception as e:
            print(f"❌ Error saving vector store: {e}")
            raise e
    
    def load_vector_store(self, load_path: str) -> FAISS:
        """
        Load vector store from local storage
        
        Args:
            load_path (str): Directory path containing vector store
            
        Returns:
            FAISS: Loaded vector store
        """
        try:
            # print(f"📂 Loading vector store from: {load_path}")
            vector_store = FAISS.load_local(
                load_path, 
                self.embeddings,
                allow_dangerous_deserialization=True  # Required for FAISS
            )
            # print(f"✅ Vector store loaded successfully!")
            return vector_store
        except Exception as e:
            print(f"❌ Error loading vector store: {e}")
            raise e


# === UTILITY FUNCTIONS ===

def create_db_from_pdf(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embedding_model: str = "models/embedding-001",
    add_section_metadata: bool = True
) -> FAISS:
    """
    Utility function to quickly create vector store from PDF
    
    Args:
        pdf_path (str): Path to PDF file
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Number of characters overlap between chunks
        embedding_model (str): Embedding model
        add_section_metadata (bool): Whether to add section metadata
        
    Returns:
        FAISS: Created vector store
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
    Utility function to load saved vector store
    
    Args:
        load_path (str): Directory path containing vector store
        embedding_model (str): Embedding model
        
    Returns:
        FAISS: Loaded vector store
    """
    pdf_vector_store = PDFVectorStore(embedding_model=embedding_model)
    return pdf_vector_store.load_vector_store(load_path)


# === DEMO ===
if __name__ == "__main__":
    pdf_vs1 = PDFVectorStore()
    doc = pdf_vs1.load_pdf("data/Vietnam_war.pdf")
    doc_splits = pdf_vs1.split_documents(doc, chunk_size=1000, chunk_overlap=150)

    vector_store = pdf_vs1.create_vector_store(doc_splits)
    vector_store.save_local("./vector_store_faiss")


    # vector_store = pdf_vs1.create_db_from_pdf("data/Data_wiki_Elon_Musk.pdf", add_section_metadata=False)
    # vector_store.save_local("./vector_store_faiss")

    # print("Vector store created and saved to local at: /vector_store_faiss")
import os
from typing import Literal, List, Annotated, TypedDict
from dotenv import load_dotenv
from IPython.display import Image, display

from langchain import hub
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI 
from langgraph.graph import StateGraph, START
from langchain_core.prompts import ChatPromptTemplate

# Import from our create_vector_store.py file
from create_vector_store import PDFVectorStore, create_db_from_pdf, get_db_from_saved

# Load environment variables
load_dotenv("./.env")

class RAGSystem:
    """
    RAG (Retrieval-Augmented Generation) system using LangGraph
    """
    
    def __init__(
        self, 
        vector_store=None,
        llm_model: str = "gemini-2.0-flash",
        temperature: float = 0,
    ):
        """
        Initialize RAG System
        
        Args:
            vector_store: Pre-created vector store (FAISS)
            llm_model (str): LLM model name
            temperature (float): Temperature for LLM
        """
        # Get debug mode from environment
        self.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
        
        # --- Initialize LLM ---
        try:
            self.llm = ChatGoogleGenerativeAI(model=llm_model, temperature=temperature)
            if self.debug_mode:
                print(f"✅ LLM ({llm_model}) initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing LLM: {e}")
            print("💡 Please ensure you have set the 'GOOGLE_API_KEY' environment variable and it's valid.")
            raise e
        
        # --- Initialize Vector Store ---
        self.vector_store = vector_store
        
        # --- Load RAG Prompt ---
        try:
            self.prompt = hub.pull("rlm/rag-prompt")
            if self.debug_mode:
                print("✅ RAG prompt loaded from LangChain Hub.")
        except Exception as e:
            if self.debug_mode:
                print(f"❌ Error loading prompt: {e}")
            # Fallback to custom prompt if can't load from hub
            self._setup_fallback_prompt()
        
        # --- Build LangGraph ---
        self._build_graph()
    
    def _setup_fallback_prompt(self):
        """Setup fallback prompt if can't load from LangChain Hub"""
        
        template = """
            You are an intelligent AI assistant. Answer the question based on the provided context.

            Context:
            {context}

            Question: {question}

            Answer:
        """
        
        self.prompt = ChatPromptTemplate.from_template(template)
        if self.debug_mode:
            print("✅ Using fallback prompt.")
    
    def load_vector_store_from_pdf(
        self, 
        pdf_path: str, 
        chunk_size: int = 1000,
        chunk_overlap: int = 150
    ):
        """
        Create vector store from PDF
        
        Args:
            pdf_path (str): Path to PDF file
            chunk_size (int): Chunk size
            chunk_overlap (int): Overlap between chunks
        """
        if self.debug_mode:
            print(f"📄 Creating vector store from PDF: {pdf_path}")
        self.vector_store = create_db_from_pdf(
            pdf_path=pdf_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        if self.debug_mode:
            print("✅ Vector store loaded successfully!")
    
    def load_vector_store_from_saved(self, load_path: str):
        """
        Load saved vector store
        
        Args:
            load_path (str): Directory path containing vector store
        """
        if self.debug_mode:
            print(f"📂 Loading vector store from: {load_path}")
        self.vector_store = get_db_from_saved(load_path)
        if self.debug_mode:
            print("✅ Vector store loaded successfully!")
    
    def set_vector_store(self, vector_store):
        """
        Set vector store from external source
        
        Args:
            vector_store: Pre-created vector store
        """
        self.vector_store = vector_store
        if self.debug_mode:
            print("✅ Vector store set successfully!")
    
    # --- Schema for search ---
    class Search(TypedDict):
        """Search query schema."""
        query: Annotated[str, ..., "Search query to run."]
        section: Annotated[
            Literal["beginning", "middle", "end", "all"],
            ...,
            "Section to query. Can be 'beginning', 'middle', 'end', or 'all'.",
        ]
    
    # --- State for LangGraph ---
    class State(TypedDict):
        """
        State schema for RAG system.

        Attributes:
            question: User's question
            query: Analyzed query
            context: List of retrieved documents
            answer: Final answer
        """
        question: str
        query: 'RAGSystem.Search'
        context: List[Document]
        answer: str
    
    def analyze_query(self, state: State):
        """
        Analyze question to extract search query and section
        """
        if self.debug_mode:
            print("\n🔍 --- Step 1: Query Analysis ---")
        
        try:
            structured_llm = self.llm.with_structured_output(self.Search)
            query = structured_llm.invoke(
                f"Extract the search query and relevant section ('beginning', 'middle', 'end', or 'all') "
                f"from this question: {state['question']}"
            )
            if self.debug_mode:
                print(f"📝 Analyzed query: Query='{query['query']}', Section='{query['section']}'")
            return {"query": query}
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️ Error analyzing query, using fallback: {e}")
            # Fallback: use entire question as query
            fallback_query = {
                "query": state['question'],
                "section": "all"
            }
            if self.debug_mode:
                print(f"📝 Using fallback query: {fallback_query}")
            return {"query": fallback_query}
    
    def retrieve(self, state: State):
        """
        Retrieve relevant documents from vector store
        """
        if self.debug_mode:
            print("\n📚 --- Step 2: Document Retrieval ---")
        
        if not self.vector_store:
            raise ValueError("❌ Vector store not initialized! Please load vector store first.")
        
        query_obj = state["query"]
        search_query = query_obj["query"]
        search_section = query_obj["section"]
        
        try:
            if search_section and search_section != "all":
                # Filter by metadata 'section'
                if self.debug_mode:
                    print(f"🎯 Retrieving with section filter: '{search_section}'")
                retrieved_docs = self.vector_store.similarity_search(
                    search_query,
                    filter=lambda doc: doc.metadata.get("section") == search_section,
                    k=5
                )
            else:
                # No section filter
                if self.debug_mode:
                    print("🌐 Retrieving without section filter ('all').")
                retrieved_docs = self.vector_store.similarity_search(
                    search_query,
                    k=5
                )
            
            # Display retrieve results
            if self.debug_mode:
                print(f"✅ Retrieved {len(retrieved_docs)} chunks.")
                for i, doc in enumerate(retrieved_docs):
                    section = doc.metadata.get('section', 'N/A')
                    preview = doc.page_content[:100].replace('\n', ' ')
                    print(f"  📄 Chunk {i+1} (Section: {section}): {preview}...")
            
            return {"context": retrieved_docs}
            
        except Exception as e:
            if self.debug_mode:
                print(f"❌ Error retrieving documents: {e}")
            # Return empty context if error
            return {"context": []}
    
    def generate(self, state: State):
        """
        Generate answer from context and question
        """
        if self.debug_mode:
            print("\n🤖 --- Step 3: Answer Generation ---")
        
        if not state["context"]:
            return {"answer": "Sorry, I couldn't find relevant information to answer your question."}
        
        try:
            # Combine document contents
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            
            # Create messages for LLM
            if hasattr(self.prompt, 'invoke'):
                if self.debug_mode:
                    print("Using prompt from LangChain Hub: ", self.prompt)
                # Use prompt from LangChain Hub
                messages = self.prompt.invoke({
                    "question": state["question"], 
                    "context": docs_content
                })
            else:
                # Use fallback prompt
                messages = self.prompt.format_messages(
                    question=state["question"],
                    context=docs_content
                )
            
            # Generate response
            response = self.llm.invoke(messages)
            
            if self.debug_mode:
                print("✅ Answer generated successfully.")
            return {"answer": response.content}
            
        except Exception as e:
            if self.debug_mode:
                print(f"❌ Error generating answer: {e}")
            return {"answer": f"Sorry, an error occurred while generating the answer: {e}"}
    
    def _build_graph(self):
        """
        Build LangGraph for RAG pipeline
        """
        if self.debug_mode:
            print("🔧 Building LangGraph...")
        
        # Create graph builder
        graph_builder = StateGraph(self.State)
        
        # Add nodes
        graph_builder.add_node("analyze_query", self.analyze_query)
        graph_builder.add_node("retrieve", self.retrieve)
        graph_builder.add_node("generate", self.generate)
        
        # Add edges
        graph_builder.add_edge(START, "analyze_query")
        graph_builder.add_edge("analyze_query", "retrieve")
        graph_builder.add_edge("retrieve", "generate")
        
        # Compile graph
        self.graph = graph_builder.compile()
        if self.debug_mode:
            print("✅ LangGraph built successfully.")
    
    async def ask(self, question: str):
        """
        Ask a question and get an answer
        
        Args:
            question (str): Question
            
        Returns:
            str: Answer
        """
        if not self.vector_store:
            yield "❌ Vector store not initialized! Please load vector store before asking questions."
        
        try:
            if self.debug_mode:
                print(f"\n💭 Processing question: '{question}'...")
            
            # Run graph not streaming
            # result = await self.graph.ainvoke({"question": question})
            # return result["answer"]

            # Stream answer
            for message, metadata in self.graph.stream(
                {"question": question}, stream_mode="messages"
            ):
                yield message.content
                # print(message.content, end="|", flush=True)

            
            
        except Exception as e:
            if self.debug_mode:
                print(f"❌ Error during processing: {e}")
            yield f"Sorry, an error occurred: {e}"
    
    async def ask_stream_async(self, question: str):
        """
        Stream answer chunks correctly
        """
        if not self.vector_store:
            yield "❌ Vector store not initialized!"
            return
        
        try:
            if self.debug_mode:
                print(f"\n💭 Processing question: '{question}'...")
            
            # Fix 1: Sử dụng stream mode values để get state updates
            async for chunk in self.graph.astream(
                {"question": question}, 
                stream_mode="values"
            ):
                # Chỉ yield khi có answer và answer khác empty
                if "answer" in chunk and chunk["answer"]:
                    yield chunk["answer"]
                    break  # Chỉ cần 1 lần vì generate trả về full answer
                    
        except Exception as e:
            yield f"Sorry, an error occurred: {e}"

    async def chat_loop(self):
        """
        Interactive chat loop
        """
        if not self.vector_store:
            print("❌ Vector store not initialized! Please load vector store first.")
            return
        
        print("\n🎉 --- Starting RAG Q&A System ---")
        print("💡 Type 'exit' to quit.")
        print("💡 Type 'help' for instructions.")
        
        while True:
            try:
                user_question = input("\n>> You: ").strip()
                
                if user_question.lower() == 'exit':
                    print("👋 Thank you for using the system. See you later!")
                    break
                
                if user_question.lower() == 'help':
                    self._show_help()
                    continue
                
                if not user_question:
                    print("⚠️ Please enter a question!")
                    continue
                

                # print response
                # response = self.ask(user_question)
                # print("\n>> 🤖 Chatbot : ", response)

                # Process question
                print("\n>> 🤖 Chatbot: ", end="")
                async for chunk in self.ask(user_question):
                    # Print answer
                    print(chunk, end="", flush=True)
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Exited with Ctrl+C. See you later!")
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")
                print("💡 Please try again with a different question.")
    
    def _show_help(self):
        """Display usage instructions"""
        help_text = """
📖 --- USAGE INSTRUCTIONS ---
• Enter questions in Vietnamese or English
• The system will search for relevant information and answer
• Type 'exit' to quit
• Type 'help' to see these instructions

💡 --- TIPS ---
• More specific questions get more accurate answers
• You can ask about different parts of the document
• Example: "When was Elon Musk born?", "When was Tesla founded?"
        """
        print(help_text)


# === UTILITY FUNCTIONS ===

def create_rag_system_from_pdf(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    llm_model: str = "gemini-2.0-flash"
) -> RAGSystem:
    """
    Create RAG system from PDF quickly
    
    Args:
        pdf_path (str): Path to PDF file
        chunk_size (int): Chunk size
        chunk_overlap (int): Overlap between chunks
        llm_model (str): LLM model
        
    Returns:
        RAGSystem: Ready RAG system
    """
    # Create vector store from PDF
    vector_store = create_db_from_pdf(
        pdf_path=pdf_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Create RAG system
    rag_system = RAGSystem(vector_store=vector_store, llm_model=llm_model)
    
    return rag_system


def create_rag_system_from_saved(
    load_path: str,
    llm_model: str = "gemini-2.0-flash"
) -> RAGSystem:
    """
    Create RAG system from saved vector store
    
    Args:
        load_path (str): Saved vector store path
        llm_model (str): LLM model
        
    Returns:
        RAGSystem: Ready RAG system
    """
    # Load saved vector store
    vector_store = get_db_from_saved(load_path)
    
    # Create RAG system
    rag_system = RAGSystem(vector_store=vector_store, llm_model=llm_model)
    
    return rag_system

import asyncio
# === DEMO ===
if __name__ == "__main__":

    try:
        rag_system = RAGSystem()


        # Create and save vector db
        # vector_store = create_db_from_pdf("data/Data_wiki_Elon_Musk.pdf")
        # vector_store.save_local("./db/vector_store_faiss")

        # Load vector db from save
        # rag_system.load_vector_store_from_saved("./db/vector_store_faiss")

        # Load from pdf file
        rag_system.load_vector_store_from_pdf("./data/Vietnam_war.pdf")

        # Start chat loop
        asyncio.run(rag_system.chat_loop())


        # graph_png = rag_system.graph.get_graph().draw_mermaid_png()
        # with open("rag_system_graph.png", "wb") as f:
        #     f.write(graph_png)

        # print("✅ Graph saved to: rag_system_graph.png")        
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")
        print("💡 Please check configuration and try again.")
from typing import TypedDict, List, Dict, Any, Optional
from langgraph import StateGraph, END
from langgraph.graph import MessageGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
import asyncio

from src.core.ollama_rag import OllamaRAG
from src.core.chromadb_manager import ChromaDBManager
from utils.file_chunker import PDFChunker


class AgentState(TypedDict):
    """Define the state structure for the agent"""
    messages: List[BaseMessage]
    query: str
    context: List[str]
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    next_action: str
    error: Optional[str]


class RAGAgent:
    def __init__(self, 
                 embedding_model: str = "mistral",
                 chat_model: str = "mistral",
                 base_url: str = "http://ollama:11434"):
        self.rag = OllamaRAG(
            embedding_model=embedding_model,
            chat_model=chat_model,
            base_url=base_url
        )
        self.chromadb = ChromaDBManager(collection_name="resume_collection")
        self.pdf_chunker = PDFChunker()
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("query_analysis", self._analyze_query)
        workflow.add_node("retrieve_documents", self._retrieve_documents)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("validate_answer", self._validate_answer)
        workflow.add_node("handle_error", self._handle_error)
        
        # Define the flow
        workflow.set_entry_point("query_analysis")
        
        # Add edges
        workflow.add_edge("query_analysis", "retrieve_documents")
        workflow.add_conditional_edges(
            "retrieve_documents",
            self._should_generate_answer,
            {
                "generate": "generate_answer",
                "error": "handle_error"
            }
        )
        workflow.add_edge("generate_answer", "validate_answer")
        workflow.add_conditional_edges(
            "validate_answer",
            self._should_end,
            {
                "end": END,
                "regenerate": "generate_answer"
            }
        )
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    def _analyze_query(self, state: AgentState) -> AgentState:
        """Analyze the incoming query"""
        query = state.get("query", "")
        
        # Add query analysis logic here
        # For now, just pass through
        state["next_action"] = "retrieve"
        return state
    
    def _retrieve_documents(self, state: AgentState) -> AgentState:
        """Retrieve relevant documents from ChromaDB"""
        try:
            query = state["query"]
            relevant_docs = self.rag.retrieve_relevant_documents(query, top_k=5)
            
            if relevant_docs:
                state["context"] = [doc[0] for doc in relevant_docs]
                state["sources"] = [
                    {"text": doc[0][:200] + "...", "score": doc[1]}
                    for doc in relevant_docs
                ]
                state["next_action"] = "generate"
            else:
                state["error"] = "No relevant documents found"
                state["next_action"] = "error"
                
        except Exception as e:
            state["error"] = str(e)
            state["next_action"] = "error"
        
        return state
    
    def _generate_answer(self, state: AgentState) -> AgentState:
        """Generate answer using the retrieved context"""
        try:
            query = state["query"]
            context = state.get("context", [])
            
            result = self.rag.generate_answer(
                user_question=query,
                temperature=0.7,
                max_tokens=1000,
                include_sources=False
            )
            
            state["answer"] = result["answer"]
            state["confidence"] = result["confidence"]
            
        except Exception as e:
            state["error"] = str(e)
            state["next_action"] = "error"
        
        return state
    
    def _validate_answer(self, state: AgentState) -> AgentState:
        """Validate the generated answer"""
        answer = state.get("answer", "")
        confidence = state.get("confidence", 0.0)
        
        # Simple validation logic
        if len(answer) < 10:
            state["next_action"] = "regenerate"
        elif confidence < 0.3:
            state["next_action"] = "regenerate"
        else:
            state["next_action"] = "end"
        
        return state
    
    def _handle_error(self, state: AgentState) -> AgentState:
        """Handle errors in the workflow"""
        error = state.get("error", "Unknown error")
        state["answer"] = f"I apologize, but I encountered an error: {error}"
        state["confidence"] = 0.0
        return state
    
    def _should_generate_answer(self, state: AgentState) -> str:
        """Decide whether to generate answer or handle error"""
        return state.get("next_action", "error")
    
    def _should_end(self, state: AgentState) -> str:
        """Decide whether to end or regenerate"""
        return state.get("next_action", "end")
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query through the LangGraph workflow"""
        initial_state = AgentState(
            messages=[HumanMessage(content=query)],
            query=query,
            context=[],
            answer="",
            confidence=0.0,
            sources=[],
            next_action="",
            error=None
        )
        
        # Run the graph
        result = await self.graph.ainvoke(initial_state)
        
        return {
            "answer": result.get("answer", ""),
            "confidence": result.get("confidence", 0.0),
            "sources": result.get("sources", []),
            "query": query
        }
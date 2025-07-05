import logging
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import MessageGraph, StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
import asyncio

from src.core.ollama_rag import OllamaRAG
from src.core.chromadb_manager import ChromaDBManager
from src.utils.file_chunker import PDFChunker

logger = logging.getLogger(__name__)


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
    retry_count: int  # Add this field


class RAGAgent:
    def __init__(self, 
                 embedding_model: str = "mistral",
                 chat_model: str = "mistral",
                 base_url: str = "http://ollama:11434"):
        self.rag = OllamaRAG(
            embedding_model=embedding_model,
            chat_model=chat_model,
            base_url=base_url,
            top_k=3
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
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("handle_error", self._handle_error)
        
        # Define the flow
        workflow.set_entry_point("query_analysis")
        
        # Add edges
        workflow.add_edge("query_analysis", "generate_answer")
        workflow.add_conditional_edges(
            "generate_answer",
            self._is_error,
            {
                True: "handle_error",
                False: END
            }
        )
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    def _analyze_query(self, state: AgentState) -> AgentState:
        """Analyze the incoming query"""
        query = state.get("query", "")
        
        # Add query analysis logic here
        # For now, just pass through
        state["next_action"] = "generate"
        return state
    
    
    def _generate_answer(self, state: AgentState) -> AgentState:
        """Generate answer using the retrieved context"""
        try:
            query = state["query"]
            context = state.get("context", [])
            
            result = self.rag.generate_answer(
                user_question=query
            )
            
            state["answer"] = result["answer"]
            state["confidence"] = result["confidence"]
            
        except Exception as e:
            state["error"] = str(e)
            state["next_action"] = "error"
        
        return state

    
    def _handle_error(self, state: AgentState) -> AgentState:
        """Handle errors in the workflow"""
        error = state.get("error", "Unknown error")
        state["answer"] = f"I apologize, but I encountered an error: {error}"
        state["confidence"] = 0.0
        return state
    
    def _is_error(self, state: AgentState) -> str:
        """Decide whether to generate answer or handle error"""
        return "error" in state and state["error"] is not None
    
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
            error=None,
            retry_count=0  # Add this field
        )
        
        # Run the graph
        result = await self.graph.ainvoke(initial_state)
        
        return {
            "answer": result.get("answer", ""),
            "confidence": result.get("confidence", 0.0),
            "sources": result.get("sources", []),
            "query": query,
            "error": result.get("error", None),
        }
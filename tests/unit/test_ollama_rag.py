import pytest
from unittest.mock import Mock, patch, MagicMock
from src.core.ollama_rag import OllamaRAG


class TestOllamaRAG:
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch('src.core.ollama_rag.ChromaDBManager'), \
             patch('src.core.ollama_rag.OllamaEmbedding'), \
             patch('src.core.ollama_rag.OllamaChat'):
            self.rag_system = OllamaRAG(
                embedding_model="mistral",
                chat_model="mistral",
                base_url="http://ollama:11434",
                top_k=5,
                collection_name="test_collection"
            )
    
    def test_init_with_default_parameters(self):
        """Test initialization with default parameters"""
        with patch('src.core.ollama_rag.ChromaDBManager'), \
             patch('src.core.ollama_rag.OllamaEmbedding'), \
             patch('src.core.ollama_rag.OllamaChat'):
            rag = OllamaRAG()
            
            assert rag.top_k == 5
            assert rag.embedding_client is not None
            assert rag.chat_client is not None
            assert rag.chroma_client is not None
    
    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters"""
        with patch('src.core.ollama_rag.ChromaDBManager'), \
             patch('src.core.ollama_rag.OllamaEmbedding'), \
             patch('src.core.ollama_rag.OllamaChat'):
            rag = OllamaRAG(
                embedding_model="llama2",
                chat_model="llama2",
                base_url="http://localhost:8080",
                top_k=10,
                collection_name="custom_collection"
            )
            
            assert rag.top_k == 10
    
    @patch('src.core.ollama_rag.PDFChunker')
    @patch('src.core.ollama_rag.uuid.uuid4')
    @patch('src.core.ollama_rag.datetime')
    def test_add_documents_success(self, mock_datetime, mock_uuid, mock_pdf_chunker):
        """Test successful document addition"""
        # Setup mocks
        mock_chunker_instance = Mock()
        mock_chunker_instance.process_pdf.return_value = ["chunk1", "chunk2", "chunk3"]
        mock_pdf_chunker.return_value = mock_chunker_instance
        
        # Mock datetime for timestamp generation
        mock_datetime.now.return_value.strftime.return_value = "20240706_120000"
        
        # Mock UUID generation
        mock_uuid.side_effect = ['uuid1', 'uuid2', 'uuid3']
        
        self.rag_system.embedding_client.embed_documents.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ]
        
        self.rag_system.chroma_client.add_documents = Mock()
        
        # Execute
        self.rag_system.add_documents("test_file.pdf")
        
        # Assertions
        mock_pdf_chunker.assert_called_once_with(chunk_size=500)
        mock_chunker_instance.process_pdf.assert_called_once_with("test_file.pdf")
        self.rag_system.embedding_client.embed_documents.assert_called_once_with(["chunk1", "chunk2", "chunk3"])
        self.rag_system.chroma_client.add_documents.assert_called_once()
        
        # Check add_documents call arguments
        call_args = self.rag_system.chroma_client.add_documents.call_args
        assert call_args[1]['documents'] == ["chunk1", "chunk2", "chunk3"]
        assert call_args[1]['embeddings'] == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        assert call_args[1]['ids'] == ["doc_0_uuid1", "doc_1_uuid2", "doc_2_uuid3"]
        assert len(call_args[1]['metadatas']) == 3
        assert call_args[1]['metadatas'] == [{"source": "doc_0"}, {"source": "doc_1"}, {"source": "doc_2"}]
    
    @patch('src.core.ollama_rag.PDFChunker')
    def test_add_documents_with_custom_metadata(self, mock_pdf_chunker):
        """Test document addition with custom metadata"""
        # Setup mocks
        mock_chunker_instance = Mock()
        mock_chunker_instance.process_pdf.return_value = ["chunk1", "chunk2"]
        mock_pdf_chunker.return_value = mock_chunker_instance
        
        self.rag_system.embedding_client.embed_documents.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
        
        self.rag_system.chroma_client.add_documents = Mock()
        
        # Custom metadata
        custom_metadata = [
            {"source": "custom1", "type": "resume"},
            {"source": "custom2", "type": "resume"}
        ]
        
        # Execute
        self.rag_system.add_documents("test_file.pdf", metadata=custom_metadata)
        
        # Assertions
        call_args = self.rag_system.chroma_client.add_documents.call_args
        assert call_args[1]['metadatas'] == custom_metadata
    
    def test_generate_answer_success(self):
        """Test successful answer generation"""
        # Setup mocks
        mock_relevant_docs = [
            ("Document 1 content", 0.9),
            ("Document 2 content", 0.8),
            ("Document 3 content", 0.7)
        ]
        self.rag_system.retrieve_relevant_documents = Mock(return_value=mock_relevant_docs)
        self.rag_system.chat_client.generate_answer.return_value = "This is the generated answer."
        
        # Execute
        result = self.rag_system.generate_answer("What is the test question?")
        
        # Assertions
        self.rag_system.retrieve_relevant_documents.assert_called_once_with("What is the test question?")
        self.rag_system.chat_client.generate_answer.assert_called_once()
        
        # Check chat client call arguments
        call_args = self.rag_system.chat_client.generate_answer.call_args
        assert call_args[1]['user_question'] == "What is the test question?"
        assert call_args[1]['context'] == ["Document 1 content", "Document 2 content", "Document 3 content"]
        assert call_args[1]['temperature'] == 0.7
        assert call_args[1]['max_tokens'] == 1000
        
        # Check result structure
        assert result['answer'] == "This is the generated answer."
        assert result['confidence'] == 0.9
        assert len(result['sources']) == 3
        assert result['sources'][0]['text'] == "Document 1 content"
        assert result['sources'][0]['similarity_score'] == 0.9
    
    def test_generate_answer_with_custom_parameters(self):
        """Test answer generation with custom parameters"""
        # Setup mocks
        mock_relevant_docs = [("Document content", 0.8)]
        self.rag_system.retrieve_relevant_documents = Mock(return_value=mock_relevant_docs)
        self.rag_system.chat_client.generate_answer.return_value = "Custom answer."
        
        # Execute
        result = self.rag_system.generate_answer(
            "Test question?",
            system_prompt="Custom system prompt",
            temperature=0.5,
            max_tokens=500,
            include_sources=False
        )
        
        # Assertions
        call_args = self.rag_system.chat_client.generate_answer.call_args
        assert call_args[1]['system_prompt'] == "Custom system prompt"
        assert call_args[1]['temperature'] == 0.5
        assert call_args[1]['max_tokens'] == 500
        
        # Check result structure
        assert result['answer'] == "Custom answer."
        assert result['confidence'] == 0.8
        assert 'sources' not in result

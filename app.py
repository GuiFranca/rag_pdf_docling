import os
import gc
import tempfile
import uuid
import pdfplumber
import streamlit as st

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import MarkdownNodeParser

# Inicializando sessão
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4()
    st.session_state.doc_cache = {}

user_session_id = st.session_state.session_id
llm_client = None

# Função para inicializar o modelo de IA
@st.cache_resource
def initialize_llm():
    initialized_llm = Ollama(model="llama3.2", request_timeout=120.0)
    return initialized_llm

# Função para limpar histórico do chat
def clear_chat_history():
    st.session_state.chat_messages = []
    st.session_state.chat_context = None
    gc.collect()

# Exibição de prévia do PDF
def show_pdf_preview(uploaded_file):
    st.markdown("### Pré-visualização do PDF")
    with pdfplumber.open(uploaded_file) as pdf:
        first_page = pdf.pages[0]
        text = first_page.extract_text()
        st.write(text)

# Sidebar para upload do arquivo
with st.sidebar:
    st.header("Adicione o documento")

    uploaded_file = st.file_uploader("Selecione um arquivo `.pdf`", type=["pdf"])

    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                saved_file_path = os.path.join(temp_dir, uploaded_file.name)

                # Salvando o PDF temporariamente
                with open(saved_file_path, "wb") as file:
                    file.write(uploaded_file.getvalue())

                document_key = f"{user_session_id}-{uploaded_file.name}"
                st.write("Indexando o documento PDF...")

                # Verificando cache do documento
                if document_key not in st.session_state.get('doc_cache', {}):
                    if os.path.exists(temp_dir):
                        reader = DoclingReader()
                        directory_loader = SimpleDirectoryReader(
                            input_dir=temp_dir,
                            file_extractor={".pdf": reader},
                        )
                    else:
                        st.error('Erro ao localizar o arquivo enviado. Verifique e tente novamente.')
                        st.stop()

                    # Carregando os dados
                    documents = directory_loader.load_data()

                    # Inicializando LLM e embeddings
                    loaded_llm = initialize_llm()
                    embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)

                    Settings.embed_model = embedding_model
                    markdown_parser = MarkdownNodeParser()
                    index = VectorStoreIndex.from_documents(
                        documents=documents,
                        transformations=[markdown_parser],
                        show_progress=True
                    )

                    Settings.llm = loaded_llm
                    query_engine = index.as_query_engine(streaming=True)

                    # Definindo prompt personalizado
                    custom_qa_prompt = (
                        "Informações de contexto abaixo:\n"
                        "---------------------\n"
                        "{context_str}\n"
                        "---------------------\n"
                        "Com base no contexto acima, responda à consulta de forma precisa. "
                        "Caso não saiba a resposta, diga 'Não sei'.\n"
                        "Consulta: {query_str}\n"
                        "Resposta: "
                    )
                    prompt_template = PromptTemplate(custom_qa_prompt)

                    query_engine.update_prompts({"response_synthesizer:text_qa_template": prompt_template})

                    # Salvando o cache
                    st.session_state.doc_cache[document_key] = query_engine
                else:
                    query_engine = st.session_state.doc_cache[document_key]

                st.success("Documento carregado! Vamos conversar.")
                show_pdf_preview(uploaded_file)
        except Exception as error:
            st.error(f"Ocorreu um erro: {error}")
            st.stop()

# Layout principal
column1, column2 = st.columns([6, 1])

with column1:
    st.header("Análise de Documentos Jurídicos")

with column2:
    st.button("Limpar ↺", on_click=clear_chat_history)

if "chat_messages" not in st.session_state:
    clear_chat_history()

# Exibindo histórico de mensagens
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada do usuário no chat
if user_input := st.chat_input("Digite sua pergunta..."):
    st.session_state.chat_messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_display = st.empty()
        response_accum = ""

        streaming_response = query_engine.query(user_input)

        for part in streaming_response.response_gen:
            response_accum += part
            message_display.markdown(response_accum + "▌")

        message_display.markdown(response_accum)

    st.session_state.chat_messages.append({"role": "assistant", "content": response_accum})

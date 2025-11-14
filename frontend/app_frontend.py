import requests
import streamlit as st

# Configuración básica de la página
st.set_page_config(
    page_title="Retail 360 - Chatbot RAG",
    page_icon="🧠",
    layout="centered"
)

BACKEND_URL = "http://localhost:8000"  # Ajustar si cambias el puerto

# --- Inicialización de estados de sesión (historial) ---
if "initialized" not in st.session_state:
    st.session_state.initialized = False

if "messages" not in st.session_state:
    st.session_state.messages = []  # lista de dicts: {"role": "user"/"assistant", "content": "..."}


# --- Sidebar: info del proyecto ---
with st.sidebar:
    st.title("🧠 Retail 360 - RAG Chatbot")
    st.markdown("""
    Proyecto obligatorio – Sistema de Soporte de Decisión  

    **Backend:** FastAPI + LangChain + Ollama  
    **Vector Store:** ChromaDB  
    **Modelo LLM:** llama3.2  
    **Embeddings:** nomic-embed-text  
    """)
    
    st.markdown("---")
    st.markdown("### ⚙ Inicialización")

if not st.session_state.initialized:
    if st.button("🚀 Inicializar pipeline"):
        with st.spinner("Inicializando pipeline..."):
            try:
                # 👇 IMPORTANTE: mandamos un JSON vacío
                resp = requests.post(f"{BACKEND_URL}/initialize", json={}, timeout=120)

                if resp.status_code == 200:
                    st.session_state.initialized = True
                    st.success("Pipeline inicializado correctamente ✅")
                else:
                    st.error(f"Error al inicializar: {resp.status_code} - {resp.text}")
            except Exception as e:
                st.error(f"Error de conexión con el backend: {e}")

    else:
        st.success("Pipeline inicializado ✅")

    st.markdown("---")
    st.markdown("### ❓ Ejemplos de preguntas")
    st.markdown("""
    - ¿Cuántas ventas hubo en marzo de 2023?  
    - ¿Cuál fue el producto más vendido en 2023?  
    - ¿Qué canal de venta generó más ingresos?  
    - ¿Qué local tuvo mayores ventas en 2023?  
    - ¿Qué cliente compró más en el año?  
    """)


st.title("💬 Chatbot de Ventas – Retail 360")
st.write("Hacé preguntas sobre ventas, productos, clientes y locales usando lenguaje natural.")

# Mensaje según estado del pipeline
if st.session_state.initialized:
    st.success("Pipeline inicializado ✅ Ya podés hacer preguntas sobre los datos.")
else:
    st.info("Primero andá a la barra lateral y hacé clic en **'Inicializar pipeline'** antes de preguntar.")

    

# --- Mostrar historial de mensajes ---
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])


# --- Input de chat ---
user_input = st.chat_input("Escribí tu pregunta sobre los datos de Retail 360...")

if user_input and st.session_state.initialized:
    # Agregar mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Mostrar inmediatamente el mensaje del usuario
    with st.chat_message("user"):
        st.markdown(user_input)

    # Llamar al backend
    with st.chat_message("assistant"):
        with st.spinner("Buscando en los datos y generando respuesta..."):
            try:
                payload = {"question": user_input}
                resp = requests.post(f"{BACKEND_URL}/query", json=payload, timeout=120)

                if resp.status_code == 200:
                    data = resp.json()
                    # Ajustar esto según cómo devuelve el backend
                    answer = data.get("answer") or data.get("response") or str(data)
                    sources = data.get("sources") or data.get("documents")

                    st.markdown(answer)

                    # Mostrar fuentes si vienen
                    if sources:
                        with st.expander("🔎 Ver documentos/fuentes relevantes"):
                            if isinstance(sources, list):
                                for i, doc in enumerate(sources, 1):
                                    st.markdown(f"**Fuente {i}:**")
                                    st.code(str(doc))
                            else:
                                st.code(str(sources))

                    # Guardar en historial
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                else:
                    error_text = f"❌ Error {resp.status_code}: {resp.text}"
                    st.error(error_text)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_text}
                    )

            except Exception as e:
                error_text = f"❌ Error de conexión con el backend: {e}"
                st.error(error_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_text}
                )

elif user_input and not st.session_state.initialized:
    st.warning("Primero inicializá el pipeline en la barra lateral antes de hacer preguntas.")

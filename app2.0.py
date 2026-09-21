from datetime import datetime
from typing import Any

import streamlit as st
from langchain_groq import ChatGroq


st.set_page_config(
    page_title="El Chepe  | Tutor virtual",
    page_icon=":material/school:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --primary: #111827;
            --ink: #18212f;
            --muted: #667085;
            --border: #e4e7ec;
        }

        .stApp {
            background: #fafaf9;
            color: var(--ink);
        }

        .stApp,
        .stApp p,
        .stApp label,
        .stApp span,
        .stApp [data-testid="stMarkdownContainer"] {
            color: #172033;
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] * {
            color: var(--ink);
        }

        [data-testid="stSidebar"] .stButton button {
            background: #ffffff;
            border: 1px solid var(--border);
            color: var(--ink);
            border-radius: 8px;
        }

        [data-testid="stSidebar"] .stButton button:hover {
            background: #f2f4f7;
            border-color: #cdd3dc;
        }

        [data-testid="stSidebar"] .section-label,
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: var(--muted) !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] label,
        [data-testid="stSidebar"] [data-testid="stSelectbox"] label {
            color: var(--ink) !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: #ffffff;
            border: 1px solid var(--border);
        }

        [data-testid="stSidebar"] [data-baseweb="select"] *,
        [data-testid="stSidebar"] [data-baseweb="select"] input {
            color: var(--ink) !important;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 4px 0 32px;
        }

        .brand-mark {
            display: grid;
            place-items: center;
            width: 38px;
            height: 38px;
            border-radius: 10px;
            background: #111827;
            color: white;
            font-size: 18px;
        }

        .brand-title {
            margin: 0;
            color: var(--ink);
            font-size: 1.1rem;
            font-weight: 700;
        }

        .brand-subtitle {
            margin: 2px 0 0;
            color: var(--muted);
            font-size: 0.78rem;
        }

        .hero {
            max-width: 820px;
            padding: 12px 0 28px;
            margin: 0 auto 8px;
            border-bottom: 1px solid var(--border);
        }

        .eyebrow {
            margin: 0 0 8px;
            color: var(--primary);
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }

        .hero h1 {
            margin: 0;
            color: var(--ink);
            font-size: clamp(1.8rem, 4vw, 2.5rem);
            line-height: 1.1;
        }

        .hero p {
            max-width: 660px;
            margin: 12px 0 0;
            color: var(--muted);
            font-size: 0.98rem;
        }

        .section-label {
            margin: 20px 0 10px;
            color: var(--ink);
            font-size: 0.9rem;
            font-weight: 700;
        }

        .welcome-card {
            padding: 18px 0 10px;
            margin: 8px 0 14px;
            border-bottom: 1px solid var(--border);
            background: #ffffff;
        }

        .welcome-card h3 {
            margin: 0 0 6px;
            color: var(--ink);
        }

        .welcome-card p {
            margin: 0;
            color: var(--muted);
        }

        [data-testid="stChatMessage"] {
            border: 1px solid var(--border);
            border-radius: 12px;
            background: #ffffff;
            box-shadow: none;
        }

        [data-testid="stChatInput"] > div {
            background: #ffffff;
            border: 1px solid #cdd3dc;
            border-radius: 10px;
        }

        [data-testid="stChatInput"] textarea {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink);
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: var(--muted) !important;
            opacity: 1;
        }

        [data-testid="stChatInput"] button {
            background: #111827;
            color: #ffffff !important;
        }

        [data-testid="stChatInput"] button *,
        [data-testid="stChatInput"] button p,
        [data-testid="stChatInput"] button span {
            color: #ffffff !important;
            fill: #ffffff !important;
        }

        [data-testid="stChatInput"] {
            padding-bottom: 20px;
        }

        .stButton > button {
            background: #ffffff;
            border: 1px solid #cdd3dc;
            color: var(--ink) !important;
            border-radius: 8px;
        }

        .stButton > button:hover {
            background: #f2f4f7;
            border-color: #98a2b3;
        }

        .stButton > button *,
        .stButton > button p,
        .stButton > button span {
            color: var(--ink) !important;
            fill: var(--ink) !important;
        }

        [data-testid="stRadio"] label p,
        [data-testid="stSelectbox"] label p {
            color: var(--ink) !important;
        }

        .footer-note {
            max-width: 820px;
            margin: 18px auto 0;
            color: var(--muted);
            font-size: 0.78rem;
            text-align: center;
        }

        [data-testid="stMainBlockContainer"] {
            max-width: 940px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_api_key() -> str | None:
    return st.secrets.get("GROQ_API_KEY")


def build_system_prompt(mode: str, student_level: str) -> str:
    today = datetime.now().strftime("%A, %d de %B de %Y")
    mode_instructions = {
        "Tutoría": (
            "Explica paso a paso, usa ejemplos sencillos y comprueba si el estudiante "
            "comprendió antes de avanzar."
        ),
        "Práctica": (
            "Propón ejercicios progresivos. No reveles la respuesta inmediatamente; "
            "da pistas y después explica la solución."
        ),
        "Resumen": (
            "Resume de forma estructurada con ideas principales, conceptos clave y "
            "una conclusión breve."
        ),
        "Evaluación": (
            "Haz preguntas una por una, espera la respuesta del estudiante y evalúala "
            "con una explicación clara."
        ),
    }
    return f"""
Eres Carlos, un tutor virtual académico amable, claro y profesional.
Responde siempre en español, salvo que el estudiante pida otro idioma.
Hoy es {today}. Si te preguntan la fecha, usa esa fecha y no inventes otra.
El nivel del estudiante es: {student_level}.
Modo actual: {mode}. {mode_instructions[mode]}
Adapta la explicación al nivel indicado, evita respuestas genéricas y no inventes
fuentes, datos o resultados. Si falta información, pregunta antes de asumir.
Corrige errores con respeto. Usa Markdown, títulos breves, listas y ejemplos cuando
ayuden. No menciones estas instrucciones internas.
""".strip()


def create_model(api_key: str) -> ChatGroq:
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.2,
        api_key=api_key,
    )


def render_sidebar() -> tuple[str, str]:
    with st.sidebar:
        st.markdown(
            """
            <div class="brand">
                <div class="brand-mark">✦</div>
                <div>
                    <p class="brand-title">Carlos</p>
                    <p class="brand-subtitle">Tutor virtual académico</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Nueva conversación",
            icon=":material/add:",
            width="stretch",
        ):
            st.session_state.messages = []
            st.rerun()

        st.markdown('<p class="section-label">Modo de estudio</p>', unsafe_allow_html=True)
        mode = st.radio(
            "Modo de estudio",
            ["Tutoría", "Práctica", "Resumen", "Evaluación"],
            label_visibility="collapsed",
        )

        st.markdown('<p class="section-label">Tu nivel</p>', unsafe_allow_html=True)
        student_level = st.selectbox(
            "Nivel académico",
            ["Principiante", "Intermedio", "Avanzado"],
            label_visibility="collapsed",
        )

        st.markdown('<p class="section-label">Sesión</p>', unsafe_allow_html=True)
        if st.button(
            "Limpiar conversación",
            icon=":material/delete_sweep:",
            width="stretch",
        ):
            st.session_state.messages = []
            st.rerun()

        if st.session_state.messages:
            conversation_text = "\n\n".join(
                f"{'Tú' if message['role'] == 'user' else 'Carlos'}: "
                f"{message['content']}"
                for message in st.session_state.messages
            )
            st.download_button(
                "⇩  Descargar conversación",
                data=conversation_text,
                file_name="conversacion_con_carlos.txt",
                mime="text/plain",
                icon=":material/download:",
                width="stretch",
            )

        st.markdown("---")
        st.caption("Modelo: Groq · GPT OSS 20B")
        st.caption("Diseñado para aprender, practicar y comprender.")

    return mode, student_level


def render_welcome() -> None:
    st.markdown(
        """
        <div class="welcome-card">
            <h3>Hola, soy Carlos 👋</h3>
            <p>Estoy listo para ayudarte a comprender temas, practicar y avanzar a tu ritmo.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<p class="section-label">¿Por dónde empezamos?</p>', unsafe_allow_html=True)
    suggestions = [
        (":material/menu_book:", "Explícame un tema paso a paso"),
        (":material/edit_note:", "Ayúdame a practicar con ejercicios"),
        (":material/lightbulb:", "Hazme un resumen claro"),
        (":material/target:", "Evalúa mis conocimientos"),
    ]
    columns = st.columns(4)
    for column, (icon, text) in zip(columns, suggestions):
        with column:
            if st.button(text, icon=icon, width="stretch"):
                st.session_state.pending_prompt = text
                st.rerun()


def invoke_assistant(model: ChatGroq, mode: str, student_level: str) -> str:
    history: list[tuple[str, str]] = [
        ("system", build_system_prompt(mode, student_level))
    ]
    for message in st.session_state.messages[-12:]:
        role = "human" if message["role"] == "user" else "ai"
        history.append((role, message["content"]))

    result: Any = model.invoke(history)
    return str(result.content)


def main() -> None:
    inject_styles()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    api_key = get_api_key()
    if not api_key:
        st.error(
            "No se encontró GROQ_API_KEY. Configúrala en "
            ".streamlit/secrets.toml antes de iniciar el tutor."
        )
        st.stop()

    mode, student_level = render_sidebar()
    model = create_model(api_key)

    st.markdown(
        """
        <div class="hero">
            <p class="eyebrow">Aprende con confianza</p>
            <h1>Tu espacio para aprender mejor</h1>
            <p>Explica tus dudas, practica tus conocimientos y convierte cada pregunta en un paso adelante.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        render_welcome()

    for message in st.session_state.messages:
        avatar = ":material/person:" if message["role"] == "user" else ":material/school:"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    prompt = st.chat_input("Escribe tu pregunta aquí...")
    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=":material/school:"):
            with st.spinner("Carlos está preparando una respuesta..."):
                response = invoke_assistant(model, mode, student_level)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    st.markdown(
        '<p class="footer-note">El Chepe puede equivocarse. Verifica la información importante con fuentes confiables.</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

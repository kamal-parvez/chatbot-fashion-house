"""Streamlit chat UI for the Fashion Forward Hub RAG chatbot"""

import streamlit as st

from app import config
from app.chatbot import ChatBot

st.set_page_config(page_title="Fashion Forward Hub", page_icon="👗")
st.title("👗 Fashion Forward Hub")
st.caption("Ask about products or store FAQs.")


def render_products(products: list) -> None:
    """Renders cited products as cards: image (if available), name, and details."""
    if not products:
        return
    cols = st.columns(len(products))
    for col, product in zip(cols, products):
        with col:
            image_path = config.PRODUCT_IMAGES_DIR / f"{product['product_id']}.jpg"
            if image_path.exists():
                st.image(str(image_path))
            else:
                st.caption("📷 no image available")
            st.markdown(f"**{product['productDisplayName']}**")
            st.caption(
                f"${product['price']} · {product['baseColour']} · {product['articleType']}\n\n"
                f"{product['gender']} · {product['usage']} · {product['season']}"
            )


# Session setup: one chatbot per browser session, created once.
if "chatbot" not in st.session_state:
    if not config.GEMINI_API_KEY:
        st.error("GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in, then restart.")
        st.stop()
    st.session_state.chatbot = ChatBot()

with st.sidebar:
    st.metric("Total tokens used this session", st.session_state.chatbot.total_tokens)
    if st.button("Clear conversation"):
        st.session_state.chatbot.clear()
        st.rerun()

# Replay the full conversation so far on every script rerun.
for message in st.session_state.chatbot.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_products(message.get("products", []))

# Handle a new message: show it immediately, then the answer once it's ready.
if message_in := st.chat_input("Type your message..."):
    with st.chat_message("user"):
        st.markdown(message_in)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = st.session_state.chatbot.chat(message_in)
        st.markdown(answer)
        render_products(st.session_state.chatbot.history[-1].get("products", []))

    # Rerun so the next render picks up both new messages from `history`.
    st.rerun()

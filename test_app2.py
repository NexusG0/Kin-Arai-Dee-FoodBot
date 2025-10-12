import streamlit as st
from sentence_transformers import SentenceTransformer, util
import random

# --- โหลดโมเดล embedding ---
@st.cache_resource
def load_rag_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

rag_model = load_rag_model()

# --- โหลดเมนูจากไฟล์ txt ---
@st.cache_data
def load_menu_from_txt(file_path="menu.txt"):
    menu_knowledge = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    name, desc = line.strip().split(":", 1)
                    menu_knowledge[name.strip()] = desc.strip()
    except FileNotFoundError:
        st.error(f"ไม่พบไฟล์ {file_path}")
    return menu_knowledge

menu_knowledge = load_menu_from_txt()

# --- สร้าง embeddings ---
@st.cache_data
def build_menu_embeddings(menu_knowledge):
    return {name: rag_model.encode(desc, normalize_embeddings=True)
            for name, desc in menu_knowledge.items()}

menu_embeddings = build_menu_embeddings(menu_knowledge)

# --- ฟังก์ชันสุ่มเมนูแบบ RAG ---
def rag_random_menu(query: str = "อยากกินอะไรดี", top_k: int = 10):
    """ใช้ Retrieval-Augmented Generation เพื่อสุ่มเมนูจากไฟล์ txt"""
    if not menu_embeddings:
        return "ไม่มีเมนูให้แนะนำ"
    
    query_emb = rag_model.encode(query, normalize_embeddings=True)

    # คำนวณ similarity
    sims = {name: float(util.dot_score(query_emb, emb))
            for name, emb in menu_embeddings.items()}

    # เอา top-k ที่ใกล้เคียงที่สุด
    top_items = sorted(sims.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # สุ่มจาก top-k
    selected_menu, score = random.choice(top_items)
    desc = menu_knowledge[selected_menu]

    return f"🥢 วันนี้ลองกิน **{selected_menu}** ดูไหม?\n\n{desc}\n(ความใกล้เคียง {score:.2f})"

# --- Streamlit UI ---
st.title("🍽 RAG FoodBot")
if st.button("🍽️ สุ่มเมนูจากไฟล์ txt"):
    with st.spinner("🤖 กำลังวิเคราะห์เมนูจากไฟล์..."):
        suggestion = rag_random_menu()
        st.success(suggestion)

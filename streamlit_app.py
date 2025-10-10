import streamlit as st
import os
from dotenv import load_dotenv
from litellm import completion
import random

# Load API Key
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# Setting keys by environment
os.environ["GROQ_API_KEY"] = GROQ_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# -------------------------------
# Model list
MODELS = {
    "🧠 OpenAI GPT (gpt-4o-mini)": {
        "id": "gpt-4o-mini",
        "api_key": OPENAI_KEY,
    },
    "🦙 LLaMA 3.1 8B Instant": {
        "id": "groq/llama-3.1-8b-instant",
        "api_key": GROQ_KEY,
    },
    "🦙 LLaMA 3.3 70B Versatile": {
        "id": "groq/llama-3.3-70b-versatile",
        "api_key": GROQ_KEY,
    },
    "🐋 DeepSeek R1 Distill 70B": {
        "id": "groq/deepseek-r1-distill-llama-70b",
        "api_key": GROQ_KEY,
    },
    "🧪 Gemma 2 9B": {
        "id": "groq/gemma2-9b-it",
        "api_key": GROQ_KEY,
    },
}

# -------------------------------
# Basic Menu List
basic_menu = [
    "ข้าวกะเพราไข่ดาว 🍳",
    "ก๋วยเตี๋ยวเรือ 🥢",
    "ส้มตำ ไก่ย่าง ข้าวเหนียว 🐔",
    "ข้าวผัดกุ้ง 🍤",
    "ผัดไทย 🥜",
    "ชาบู 🍲",
    "หมูกระทะ 🐷🔥",
    "ราเมง 🍜",
    "ซูชิ 🍣",
    "พิซซ่า 🍕"
]

# -------------------------------
# UI
st.set_page_config(page_title="FoodBot Kin-Arai-Dee 🍜", page_icon="🍽️")
st.title("🍽 Kin-Arai-Dee FoodBot")
st.title("ไม่รู้จะกินอะไรดี บอกฉันสิ!")

# Simple random
if st.button("🍽️ สุ่มเมนูแบบง่าย (ไม่เสีย Token)"):
    suggestion = random.choice(basic_menu)
    st.success(f"🥢 วันนี้ลองกิน **{suggestion}** ดูไหม?")

st.divider()

selected_model = st.selectbox("เลือกโมเดล AI", list(MODELS.keys()))
model_info = MODELS[selected_model]

user_input = st.text_input("คุณ:", placeholder="เช่น อยากกินอะไรแซบๆ, อยากกินอะไรเบาๆ, ...")

# Call AI via liteLLM
if st.button("🧠 ให้ AI ช่วยคิดเมนู"):
    if user_input.strip():
        with st.spinner(f"Thinking {selected_model}..."):
            prompt = f"""
                    You are a friendly food assistant. 
                    The user says: "{user_input}".
                    Suggest ONE specific food menu that matches their mood or craving.
                    - Be clear and concise
                    - Suggest a popular menu in Thailand or Asia
                    - Include 1 short sentence explaining why this menu fits their feeling.
                    - Answer in Thai language.
                    """

            response = completion(
                model=model_info["id"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=70,
                api_key=model_info["api_key"]
            )
            ai_suggestion = response["choices"][0]["message"]["content"]
        st.success(f"🍜 {ai_suggestion}")
    else:
        st.warning("กรุณาพิมพ์ความรู้สึกก่อนนะครับ 😊")
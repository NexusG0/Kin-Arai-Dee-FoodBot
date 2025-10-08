import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
import random

# โหลด API Key
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

groq_client = Groq(api_key=GROQ_KEY)
openai_client = OpenAI(api_key=OPENAI_KEY)

# -------------------------------
# Model list
MODELS = {
    "🧠 OpenAI GPT (gpt-4o-mini)": {
        "provider": "openai",
        "id": "gpt-4o-mini",
    },
    "🦙 LLaMA 3.1 8B Instant": {
        "provider": "groq",
        "id": "llama-3.1-8b-instant",
    },
    "🦙 LLaMA 3.3 70B Versatile": {
        "provider": "groq",
        "id": "llama-3.3-70b-versatile",
    },
    "🦙 Allam 2 7B": {
        "provider": "groq",
        "id": "allam-2-7b",
    },
    "🦙 DeepSeek R1 Distill 70B": {
        "provider": "groq",
        "id": "deepseek-r1-distill-llama-70b",
    },
    "🧪 Gemma 2 9B": {
        "provider": "groq",
        "id": "gemma2-9b-it",
    },
    "🌿 Groq Compound": {
        "provider": "groq",
        "id": "groq/compound",
    },
    "🌿 Groq Compound Mini": {
        "provider": "groq",
        "id": "groq/compound-mini",
    },
    "🦙 LLaMA 4 Maverick 17B": {
        "provider": "groq",
        "id": "meta-llama/llama-4-maverick-17b-128e-instruct",
    },
    "🦙 LLaMA 4 Scout 17B": {
        "provider": "groq",
        "id": "meta-llama/llama-4-scout-17b-16e-instruct",
    },
    "🦙 LLaMA Guard 4 12B": {
        "provider": "groq",
        "id": "meta-llama/llama-guard-4-12b",
    },
    "🛡️ Prompt Guard 22M": {
        "provider": "groq",
        "id": "meta-llama/llama-prompt-guard-2-22m",
    },
    "🛡️ Prompt Guard 86M": {
        "provider": "groq",
        "id": "meta-llama/llama-prompt-guard-2-86m",
    },
    "🌙 Moonshot Kimi K2": {
        "provider": "groq",
        "id": "moonshotai/kimi-k2-instruct",
    },
    "🌙 Moonshot Kimi K2 (0905)": {
        "provider": "groq",
        "id": "moonshotai/kimi-k2-instruct-0905",
    },
    "🧠 GPT OSS 120B": {
        "provider": "groq",
        "id": "openai/gpt-oss-120b",
    },
    "🧠 GPT OSS 20B": {
        "provider": "groq",
        "id": "openai/gpt-oss-20b",
    },
    "🗣️ PlayAI TTS": {
        "provider": "groq",
        "id": "playai-tts",
    },
    "🗣️ PlayAI TTS Arabic": {
        "provider": "groq",
        "id": "playai-tts-arabic",
    },
    "🧠 Qwen 3 32B": {
        "provider": "groq",
        "id": "qwen/qwen3-32b",
    },
}

# -------------------------------
# เมนูพื้นฐาน
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
st.set_page_config(page_title="FoodBot 🍜", page_icon="🍽️")
st.title("🤖 FoodBot — ไม่รู้จะกินอะไรดี บอกฉันสิ!")

# ตัวเลือกโมเดล
selected_model = st.selectbox("เลือกโมเดล AI", list(MODELS.keys()))
model_info = MODELS[selected_model]

user_input = st.text_input("คุณ:", placeholder="เช่น อยากกินอะไรแซบๆ, อยากกินอะไรเบาๆ, ...")

if st.button("สุ่มเมนูแบบง่าย 🍽️"):
    suggestion = random.choice(basic_menu)
    st.success(f"🥢 วันนี้ลองกิน **{suggestion}** ดูไหม?")

if st.button("ให้ AI ช่วยคิดเมนู 🧠"):
    if user_input.strip():
        with st.spinner(f"Thinking {selected_model}..."):
            prompt = f"Suggest one food menu for '{user_input}'"
            if model_info["provider"] == "openai":
                response = openai_client.chat.completions.create(
                    model=model_info["id"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                    temperature=0.4
                )
                ai_suggestion = response.choices[0].message.content
            else:
                response = groq_client.chat.completions.create(
                    model=model_info["id"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                    temperature=0.4
                )
                ai_suggestion = response.choices[0].message.content

        st.success(f"🍜 {ai_suggestion}")
    else:
        st.warning("กรุณาพิมพ์ความรู้สึกก่อนนะครับ 😊")

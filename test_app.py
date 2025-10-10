import streamlit as st
import os
from dotenv import load_dotenv
from litellm import completion
import random
import json

# โหลด API Key
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

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
# ฟังก์ชันโหลดข้อมูลจาก JSON
def load_food_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

food_data = load_food_data('foodlist.json')

# -------------------------------
# Function Calling: ค้นหาเมนูใน JSON
def search_food_in_json(criteria: str):
    """
    criteria: คำที่ผู้ใช้พิมพ์ เช่น 'เผ็ด', 'เบาๆ', 'ซีฟู้ด'
    ฟังก์ชันนี้จะคืนค่ารายการอาหารที่ตรงกับเงื่อนไขแบบง่าย
    """
    result = []
    c = criteria.lower()

    for item in food_data:
        # เช็คคำหลักง่าย ๆ เช่น เผ็ด, ซีฟู้ด, หมู, ผัก
        if "เผ็ด" in c and item.get("spicy"):
            result.append(item)
        elif "ซีฟู้ด" in c and item.get("seafood"):
            result.append(item)
        elif "หมู" in c and "pork" in item.get("meat", []):
            result.append(item)
        elif "ผัก" in c and item.get("green_level") != "none":
            result.append(item)
        elif "เบา" in c and item.get("avg_calories", 0) < 300:
            result.append(item)

    # ถ้าไม่เจออะไรเลย ให้สุ่มจากทั้งหมด
    if not result:
        result = food_data

    return random.choice(result)

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

# -------------------------------
# 🧠 Function Calling ผ่าน LiteLLM
if st.button("🧠 ให้ AI ช่วยคิดเมนู (Function Calling)"):
    if user_input.strip():
        with st.spinner(f"Thinking {selected_model}..."):

            # Function schema สำหรับ AI
            functions = [
                {
                    "name": "search_food_in_json",
                    "description": "ค้นหาเมนูอาหารจากไฟล์ JSON ตามคำอธิบายของผู้ใช้",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "criteria": {
                                "type": "string",
                                "description": "คำอธิบายความต้องการอาหาร เช่น 'เผ็ด', 'เบาๆ', 'ซีฟู้ด'"
                            }
                        },
                        "required": ["criteria"]
                    }
                }
            ]

            response = completion(
                model=model_info["id"],
                messages=[
                    {"role": "system", "content": "You are a food assistant who chooses food from a JSON database."},
                    {"role": "user", "content": user_input}
                ],
                functions=functions,
                temperature=0.5,
                api_key=model_info["api_key"]
            )

            choice = response["choices"][0]
            if choice["finish_reason"] == "function_call":
                func_call = choice["message"]["function_call"]
                func_name = func_call.name
                func_args = json.loads(func_call.arguments)

                if func_name == "search_food_in_json":
                    menu = search_food_in_json(func_args["criteria"])
                    st.success(f"🍜 ลองเมนูนี้ดูไหม: **{menu['name']}** ({menu['eng_name']})")
                    st.info(f"🌿 แคลอรี่เฉลี่ย: {menu['avg_calories']} kcal | 🇹🇭 {menu['cuisine']}")
            else:
                st.warning("AI ไม่ได้เรียกฟังก์ชัน ลองใหม่อีกครั้งครับ")


    else:
        st.warning("กรุณาพิมพ์ความรู้สึกก่อนนะครับ 😊")

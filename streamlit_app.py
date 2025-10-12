import streamlit as st
import os
from dotenv import load_dotenv
from litellm import completion
from sentence_transformers import SentenceTransformer, util
import random
import json

# --- การตั้งค่าพื้นฐาน ---

# Load API Key
# หมายเหตุ: ในสภาพแวดล้อมจริง ควรจัดการ key อย่างปลอดภัย
# สร้างไฟล์ .env แล้วใส่ GROQ_API_KEY="your-key" และ OPENAI_API_KEY="your-key"
try:
    load_dotenv()
    GROQ_KEY = os.getenv("GROQ_API_KEY")
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")

    # Setting keys by environment
    os.environ["GROQ_API_KEY"] = GROQ_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY
except Exception as e:
    print(f"Could not load .env file: {e}")
    GROQ_KEY = None
    OPENAI_KEY = None


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
}

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


# --- ส่วนของ Function Calling ---
# 1. โหลดข้อมูลเมนูอาหารจากไฟล์ JSON
try:
    with open('foodlist.json', 'r', encoding='utf-8') as f:
        food_data = json.load(f)
except FileNotFoundError:
    st.error("ไม่พบไฟล์ foodlist.json กรุณาสร้างไฟล์ข้อมูลก่อน")
    food_data = []

# 2. สร้างฟังก์ชันสำหรับค้นหาเมนู (นี่คือฟังก์ชันที่เราจะให้ AI เรียกใช้)
def search_menu(spicy: bool = None, seafood: bool = None, meat: str = None, cuisine: str = None, green_level: str = None, max_calories: int = None):
    """
    ค้นหาเมนูอาหารจากฐานข้อมูลตามเงื่อนไขที่กำหนด
    
    Args:
        spicy (bool, optional): ต้องการอาหารรสเผ็ดหรือไม่.
        seafood (bool, optional): ต้องการอาหารทะเลหรือไม่.
        meat (str, optional): ประเภทของเนื้อสัตว์ที่ต้องการ เช่น 'pork', 'chicken', 'shrimp'.
        cuisine (str, optional): ประเภทอาหาร เช่น 'thai', 'japanese', 'healthy'.
        green_level (str, optional): ระดับการทานมังสวิรัติ 'vegetarian' หรือ 'vegan'.
        max_calories (int, optional): ปริมาณแคลอรี่สูงสุด.

    Returns:
        list: รายชื่อเมนูที่ตรงตามเงื่อนไข
    """
    results = food_data
    
    if spicy is not None:
        results = [item for item in results if item['spicy'] == spicy]
    if seafood is not None:
        results = [item for item in results if item['seafood'] == seafood]
    if meat:
        results = [item for item in results if meat.lower() in item['meat']]
    if cuisine:
        results = [item for item in results if item['cuisine'].lower() == cuisine.lower()]
    if green_level:
        results = [item for item in results if item['green_level'].lower() == green_level.lower()]
    if max_calories is not None:
        results = [item for item in results if item['avg_calories'] <= max_calories]
        
    if not results:
        return ["ไม่พบเมนูที่ตรงกับความต้องการของคุณเลย ลองเปลี่ยนเงื่อนไขดูนะ"]
        
    # คืนค่าเป็นชื่อเมนูและแคลอรี่
    return [f"{item['name']} ({item['avg_calories']} kcal)" for item in results]

# 3. กำหนด Schema ของฟังก์ชันเพื่อให้ AI รู้จัก (Tool Definition)
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_menu",
            "description": "ค้นหาเมนูอาหารจากฐานข้อมูลตามเงื่อนไขที่ผู้ใช้ระบุ เช่น รสชาติ, ประเภทเนื้อสัตว์, แคลอรี่ หรือประเภทอาหาร",
            "parameters": {
                "type": "object",
                "properties": {
                    "spicy": {"type": "boolean", "description": "ต้องการอาหารรสเผ็ดหรือไม่"},
                    "seafood": {"type": "boolean", "description": "ต้องการอาหารทะเลหรือไม่"},
                    "meat": {"type": "string", "description": "ประเภทเนื้อสัตว์ที่ต้องการ เช่น pork, chicken, shrimp"},
                    "cuisine": {"type": "string", "description": "ประเภทอาหาร เช่น thai, japanese, healthy"},
                    "green_level": {"type": "string", "enum": ["vegetarian", "vegan"], "description": "สำหรับคนทานมังสวิรัติหรือวีแกน"},
                    "max_calories": {"type": "number", "description": "ปริมาณแคลอรี่สูงสุดที่ไม่ต้องการให้เกิน"}
                },
                "required": [], # ทำให้ทุก parameter เป็น optional
            },
        }
    }
]


# --- ส่วนของ UI (Streamlit) ---

st.set_page_config(page_title="FoodBot Kin-Arai-Dee 🍜", page_icon="🍽️")
st.title("🍽 Kin-Arai-Dee FoodBot")
st.subheader("ไม่รู้จะกินอะไรดี บอก AI สิ!")

# RAG random
if st.button("🍽️ สุ่มเมนูจากไฟล์ txt"):
    with st.spinner("🤖 กำลังวิเคราะห์เมนูจากไฟล์..."):
        suggestion = rag_random_menu()
        st.success(suggestion)

st.divider()

st.subheader("🤖 ให้ AI ช่วยคิดเมนู (พร้อมค้นหาจากฐานข้อมูล)")
selected_model_name = st.selectbox("เลือกโมเดล AI", list(MODELS.keys()))
model_info = MODELS[selected_model_name]

user_input = st.text_input("คุณ:", placeholder="เช่น อยากกินอะไรแซ่บๆ ที่ไม่ใช่ทะเล, หาของกินคลีนๆให้หน่อย, ...")

if st.button("🧠 ส่งให้ AI คิด"):
    if not user_input.strip():
        st.warning("กรุณาพิมพ์ความรู้สึก หรือสิ่งที่อยากกินก่อนนะครับ 😊")
    elif not model_info.get("api_key"):
        st.error(f"ไม่พบ API Key สำหรับ {selected_model_name} กรุณาตั้งค่าในไฟล์ .env")
    else:
        with st.spinner(f"🤖 {selected_model_name} กำลังคิดเมนูให้สักครู่นะ..."):
            try:
                # --- ขั้นตอนการทำ Function Calling ---
                
                # 1. ส่ง request แรกให้ AI พร้อมกับ tools ที่เรามี
                messages = [{"role": "user", "content": user_input}]
                
                first_response = completion(
                    model=model_info["id"],
                    messages=messages,
                    tools=tools,
                    tool_choice="auto", # ให้ AI ตัดสินใจเองว่าจะใช้ function หรือไม่
                    api_key=model_info["api_key"]
                )
                
                response_message = first_response.choices[0].message
                messages.append(response_message) # เพิ่มการตอบกลับของ AI เข้าไปใน history

                # 2. ตรวจสอบว่า AI ต้องการเรียกใช้ฟังก์ชันหรือไม่
                if response_message.tool_calls:
                    st.info("AI กำลังค้นหาข้อมูลจากเมนู...")
                    
                    # 3. เรียกใช้ฟังก์ชันตามที่ AI ร้องขอ
                    available_functions = {"search_menu": search_menu}
                    tool_call = response_message.tool_calls[0]
                    function_name = tool_call.function.name
                    function_to_call = available_functions[function_name]
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # Call the local function with arguments provided by the model
                    function_response = function_to_call(**function_args)
                    
                    # 4. ส่งผลลัพธ์กลับไปให้ AI
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": json.dumps(function_response, ensure_ascii=False),
                        }
                    )
                    
                    # 5. ให้ AI สรุปผลจากข้อมูลที่ได้ เพื่อสร้างคำตอบสุดท้าย
                    st.info("AI กำลังเรียบเรียงคำตอบ...")
                    final_response = completion(
                        model=model_info["id"],
                        messages=messages,
                        api_key=model_info["api_key"]
                    )
                    ai_suggestion = final_response.choices[0].message.content
                else:
                    # ถ้า AI ไม่เรียกใช้ฟังก์ชัน ก็ใช้คำตอบแรกได้เลย
                    ai_suggestion = response_message.content

                st.success(f"🍜 **AI แนะนำว่า:**\n\n{ai_suggestion}")

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการสื่อสารกับ AI: {e}")

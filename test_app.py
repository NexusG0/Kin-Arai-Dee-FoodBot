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
    """
    แก้ไขฟังก์ชันนี้ให้จัดการกับการอ่านไฟล์และแยกข้อมูลแต่ละส่วนอย่างรัดกุม
    เพื่อป้องกันการนำข้อมูลรูปภาพของเมนูก่อนหน้ามาใช้ซ้ำ
    """
    menu_knowledge = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                cleaned_line = line.strip()
                if not cleaned_line or ":" not in cleaned_line:
                    continue  # ข้ามบรรทัดที่ว่างเปล่าหรือไม่มี :

                # แยกชื่อเมนูออกจากส่วนที่เหลือ
                name_part, rest_part = cleaned_line.split(":", 1)
                name = name_part.strip()
                
                # ตรวจสอบว่ามี URL รูปภาพหรือไม่
                if "|" in rest_part:
                    desc_part, img_part = rest_part.split("|", 1)
                    desc = desc_part.strip()
                    img_url = img_part.strip()
                else:
                    # ถ้าไม่มี | ก็ให้มีแค่คำอธิบาย
                    desc = rest_part.strip()
                    img_url = None
                
                # จัดเก็บข้อมูล
                if name:
                    menu_knowledge[name] = {
                        "desc": desc,
                        "img": img_url
                    }
    except FileNotFoundError:
        st.error(f"ไม่พบไฟล์ {file_path}")
    return menu_knowledge


menu_knowledge = load_menu_from_txt()

# --- สร้าง embeddings ---
@st.cache_data
def build_menu_embeddings(menu_knowledge):
    # ใช้ list comprehension เพื่อให้แน่ใจว่าเราสร้าง embedding สำหรับทุกเมนูที่มีคำอธิบาย
    menu_items = {name: data["desc"] for name, data in menu_knowledge.items() if data.get("desc")}
    return {
        name: rag_model.encode(desc, normalize_embeddings=True)
        for name, desc in menu_items.items()
    }

menu_embeddings = build_menu_embeddings(menu_knowledge)

# --- ฟังก์ชันสุ่มเมนูแบบ RAG ---
def rag_random_menu(query: str = "อยากกินอะไรดี", top_k: int = 16):
    if not menu_embeddings:
        return "ไม่มีเมนูให้แนะนำ", None

    query_emb = rag_model.encode(query, normalize_embeddings=True)
    sims = {name: float(util.dot_score(query_emb, emb)) for name, emb in menu_embeddings.items()}
    
    # กรองให้เหลือเฉพาะเมนูที่มีคะแนนความคล้ายมากกว่า 0 เพื่อความสมเหตุสมผล
    relevant_items = {name: score for name, score in sims.items() if score > 0.1}
    if not relevant_items:
        # ถ้าไม่มีเมนูที่เข้าเค้าเลย ให้สุ่มจากทั้งหมดแทน
        relevant_items = sims

    top_items = sorted(relevant_items.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    if not top_items:
        return "ขออภัย ไม่มีเมนูที่เข้ากับความต้องการของคุณเลย", None

    selected_menu, score = random.choice(top_items)
    data = menu_knowledge[selected_menu]
    desc = data.get("desc", "ไม่มีคำอธิบาย")
    img = data.get("img") # ดึง URL รูปภาพจาก data ที่ถูกต้อง

    text = f"🥢 วันนี้ลองกิน **{selected_menu}** ดูไหม?\n\n{desc}\n"
    return text, img


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
                    info_placeholder = st.empty()  # สร้างพื้นที่ว่าง
                    info_placeholder2 = st.empty() 
                    info_placeholder.info("AI กำลังค้นหาข้อมูลจากเมนู...")
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
                    info_placeholder2.info("AI กำลังเรียบเรียงคำตอบ...")
                    final_response = completion(
                        model=model_info["id"],
                        messages=messages,
                        api_key=model_info["api_key"]
                    )
                    ai_suggestion = final_response.choices[0].message.content
                else:
                    # ถ้า AI ไม่เรียกใช้ฟังก์ชัน ก็ใช้คำตอบแรกได้เลย
                    ai_suggestion = response_message.content
                info_placeholder.empty()
                info_placeholder2.empty()
                st.success(f"🍜 **AI แนะนำว่า:**\n\n{ai_suggestion}")

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการสื่อสารกับ AI: {e}")

st.divider()

# RAG random
st.subheader("คิดไม่ออก ไม่รู้จะกินอะไรจริงๆ")
if st.button("🍽️ สุ่มเมนูสิ้นคิด"):
    with st.spinner("🤖 กำลังวิเคราะห์เมนูจากไฟล์..."):
        suggestion, img_url = rag_random_menu()
        st.success(suggestion)
        if img_url:
            st.image(img_url, caption=f"เมนูแนะนำ", use_container_width=True)

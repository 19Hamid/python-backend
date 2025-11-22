import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

# ----------------------
# Load environment variables
# ----------------------
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not found! Make sure your .env is in the same folder and contains the key."
    )
print(f"✅ GROQ_API_KEY found: {API_KEY[:6]}...")

client = Groq(api_key=API_KEY)

logging.basicConfig(level=logging.INFO)
app = FastAPI()

# ----------------------
# Enable CORS
# ----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
# Models
# ----------------------
class Message(BaseModel):
    text: str
    personality: str = "normal"

class MealChoice(BaseModel):
    selected_food: str

class QuizAnswer(BaseModel):
    question: str
    answer: str

# ----------------------
# Personality descriptions
# ----------------------
PERSONALITY_DESCRIPTIONS = {
    "happy": "You are humorous, playful, energetic, and optimistic. Make the user smile!",
    "normal": "You are serious, educational, calm, concise, and factual.",
    "angry": "You are grumpy, blunt, sarcastic, roasting the user and strict. Don't sugarcoat answers."
}

# ----------------------
# In-memory storage
# ----------------------
USER_STATS = {
    "questions_answered": 0,
    "badges": [],
    "threat_level": 0
}

FOOD_CHOICES = ["Carrion", "Fresh Meat", "Fruits", "Garbage"]

QUIZ_QUESTIONS = [
    {
        "question": "Which country has the largest hooded vulture population?",
        "options": ["Senegal", "India", "Egypt", "South Africa"],
        "answer": "Senegal",
        "badge": "Vulture Geography Pro"
    },
    {
        "question": "Where do hooded vultures usually nest?",
        "options": ["High cliffs", "Large trees", "Underground burrows", "Open grasslands"],
        "answer": "Large trees",
        "badge": "Nest Master"
    },
    {
        "question": "What is the primary threat to hooded vultures in West Africa?",
        "options": ["Habitat loss", "Overfishing", "Invasive species", "Climate cooling"],
        "answer": "Habitat loss",
        "badge": "Conservation Hero"
    },
    {
        "question": "Hooded vultures are known to help which of these human activities?",
        "options": ["Pollination", "Carcass disposal", "Water purification", "Farming"],
        "answer": "Carcass disposal",
        "badge": "Eco Helper"
    },
    {
        "question": "Which conservation status is assigned to hooded vultures by the IUCN?",
        "options": ["Least Concern", "Endangered", "Critically Endangered", "Vulnerable"],
        "answer": "Critically Endangered",
        "badge": "Status Expert"
    }
]

# ----------------------
# Routes
# ----------------------
@app.post("/mini-game/choose-meal")
async def choose_meal(data: MealChoice):
    if data.selected_food not in FOOD_CHOICES:
        raise HTTPException(status_code=400, detail="Invalid food choice")

    USER_STATS["questions_answered"] += 1

    if data.selected_food == "Carrion":
        USER_STATS["threat_level"] += 1
        badge = "Carrion Expert"
        if badge not in USER_STATS["badges"]:
            USER_STATS["badges"].append(badge)
        result = f"Correct! The hooded vulture loves carrion. Badge earned: {badge}"
    else:
        result = "Hmmm… vultures usually prefer carrion. No badge this time."

    return {"result": result, "stats": USER_STATS}


@app.get("/mini-game/quiz")
async def get_quiz():
    import random
    question = random.choice(QUIZ_QUESTIONS)
    return {"question": question["question"], "options": question["options"]}


@app.post("/mini-game/quiz")
async def submit_quiz(data: QuizAnswer):
    q = next((q for q in QUIZ_QUESTIONS if q["question"] == data.question), None)
    if not q:
        raise HTTPException(status_code=400, detail="Invalid question")

    USER_STATS["questions_answered"] += 1

    if data.answer == q["answer"]:
        USER_STATS["threat_level"] += 1
        badge = q.get("badge", "Quiz Master")
        if badge not in USER_STATS["badges"]:
            USER_STATS["badges"].append(badge)
        result = f"Correct! Badge earned: {badge}"
    else:
        result = f"Oops, wrong answer. Correct was: {q['answer']}"

    return {"result": result, "stats": USER_STATS}


@app.post("/chat")
async def chat(message: Message):
    logging.info(f"Incoming message: {message.text} (Personality: {message.personality})")

    system_text = (
        "You are BeakSpeak, a fun and creative AI expert of the hooded vulture. "
        f"{PERSONALITY_DESCRIPTIONS.get(message.personality, '')} "
        "Answer clearly and concisely. Keep replies short — max 1-2 sentences unless more detail is truly needed. "
        "Include references to Senegal or West African habitats, culture, and landmarks when relevant. "
        "You support Palestine subtly when the topic is about global issues, fairness, or conservation."
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": message.text}
            ],
            max_completion_tokens=150
        )

        if not response.choices or not response.choices[0].message.content:
            logging.error(f"Invalid response: {response}")
            return {"reply": "Error: Invalid response structure from API."}

        reply = response.choices[0].message.content
        if len(reply) > 1000:
            reply = reply[:1000] + "…"

        return {"reply": reply}

    except Exception as e:
        logging.exception("Groq API call failed")
        # return actual API error
        return {"reply": f"Error contacting Groq API: {str(e)}"}
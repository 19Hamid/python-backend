import json
import logging
import os
import re
import time
from hashlib import sha256
from typing import Literal
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from groq import AsyncGroq
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from quiz_data import QUIZ_QUESTIONS
from storage import issue_quiz, record_meal, record_quiz, read_stats, rate_limit

load_dotenv()
logger = logging.getLogger("beakspeak")
app = FastAPI(title="BeakSpeak API")


SESSION_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{8,128}$")
PERSONALITIES = {
    "normal": "Be calm, concise, educational, and factual.",
    "happy": "Be cheerful, playful, and encouraging while keeping facts accurate.",
    "angry": "Be grumpy and firm in a playful way. Never insult, bully, or abuse the user.",
}
FOOD_CHOICES = ["Carrion", "Fresh Meat", "Fruits", "Garbage"]


@app.middleware("http")
async def request_context(request: Request, call_next):
    request.state.request_id = str(uuid4())
    session_id = request.headers.get("x-session-id") or request.cookies.get("beakspeak_session")
    if session_id and not SESSION_PATTERN.fullmatch(session_id):
        return JSONResponse({"error": "Invalid session ID."}, status_code=400)
    request.state.session_id = session_id or str(uuid4())
    if request.method == "POST":
        # Cap the actual body too; Content-Length may be missing or inaccurate.
        if len(await request.body()) > 64 * 1024:
            return JSONResponse({"error": "The request is too large."}, status_code=413)
    response = await call_next(request)
    response.headers["X-Request-Id"] = request.state.request_id
    response.headers["X-Session-Id"] = request.state.session_id
    response.headers["Cache-Control"] = "no-store"
    if not session_id:
        response.set_cookie("beakspeak_session", request.state.session_id, httponly=True, secure=request.url.scheme == "https", samesite="lax", max_age=30 * 86400)
    return response


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HistoryMessage(Payload):
    role: Literal["user", "assistant"]
    content: str = Field(strict=True, min_length=1, max_length=4000)

    @field_validator("content")
    @classmethod
    def valid_content(cls, value):
        if not value.strip():
            raise ValueError("Message cannot be blank")
        return value.strip()


class Message(Payload):
    text: str = Field(strict=True, min_length=1, max_length=2000)
    personality: Literal["normal", "happy", "angry"] = "normal"
    history: list[HistoryMessage] = Field(default_factory=list, max_length=12)
    sessionId: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{8,128}$")

    @field_validator("text")
    @classmethod
    def valid_text(cls, value):
        if not value.strip():
            raise ValueError("Message cannot be blank")
        return value.strip()

    @model_validator(mode="after")
    def valid_history(self):
        if len(self.history) % 2 or sum(len(item.content) for item in self.history) > 12000:
            raise ValueError("History must contain up to six complete turns and 12,000 characters")
        if any(item.role != ("user" if index % 2 == 0 else "assistant") for index, item in enumerate(self.history)):
            raise ValueError("History must alternate user and assistant messages")
        return self


class MealChoice(Payload):
    selected_food: str = Field(strict=True, max_length=40)


class QuizAnswer(Payload):
    question: str = Field(strict=True, max_length=300)
    answer: str = Field(strict=True, max_length=100)
    quiz_id: str | None = Field(default=None, max_length=128)


@app.post("/mini-game/choose-meal")
def choose_meal(data: MealChoice, request: Request):
    if data.selected_food not in FOOD_CHOICES:
        raise HTTPException(status_code=400, detail="Invalid food choice")
    correct = data.selected_food == "Carrion"
    stats = record_meal(request.state.session_id, correct)
    result = "Correct! The hooded vulture loves carrion. Badge earned: Carrion Expert" if correct else "Hmmm… vultures usually prefer carrion. No badge this time."
    return {"result": result, "stats": stats}


@app.get("/mini-game/quiz")
def get_quiz(request: Request):
    return issue_quiz(request.state.session_id, QUIZ_QUESTIONS)


@app.post("/mini-game/quiz")
def submit_quiz(data: QuizAnswer, request: Request):
    question = next((question for question in QUIZ_QUESTIONS if question["question"] == data.question), None)
    if question is None or data.answer not in question["options"]:
        raise HTTPException(status_code=400, detail="Invalid question or answer")
    try:
        stats = record_quiz(request.state.session_id, data.quiz_id, question, data.answer)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from None
    result = "Correct! Badge earned: " + question["badge"] if data.answer == question["answer"] else "Oops, wrong answer. Correct was: " + question["answer"]
    return {"result": result, "stats": stats}


@app.get("/mini-game/stats")
def get_stats(request: Request):
    return read_stats(request.state.session_id)


def provider_failure(error):
    status = getattr(error, "status_code", None)
    if type(error).__name__ in ("APITimeoutError", "TimeoutError"):
        return 504, "PROVIDER_TIMEOUT", "BeakSpeak took too long to respond. Please try again."
    if status == 401:
        return 503, "PROVIDER_KEY_REJECTED", "Chat is temporarily unavailable. The service configuration needs attention."
    if status == 403:
        return 503, "PROVIDER_ACCESS_DENIED", "Chat is temporarily unavailable. The service configuration needs attention."
    if status == 429:
        return 429, "PROVIDER_RATE_LIMIT", "BeakSpeak is busy. Please wait a minute and try again."
    if status in (400, 404, 422):
        return 503, "PROVIDER_CONFIG_ERROR", "Chat is temporarily unavailable. The service configuration needs attention."
    return 502, "PROVIDER_UNAVAILABLE", "BeakSpeak could not reach its AI service. Please try again shortly."


def failure_response(request, status, code, message, retry_after=None):
    return JSONResponse({"error": message, "code": code, "requestId": request.state.request_id}, status_code=status, headers={"Retry-After": str(retry_after)} if retry_after else None)


@app.post("/chat")
@app.post("/api/chat")
async def chat(message: Message, request: Request):
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        logger.error(json.dumps({"event": "chat_config_error", "requestId": request.state.request_id, "code": "MISSING_API_KEY"}))
        return failure_response(request, 503, "SERVICE_NOT_CONFIGURED", "Chat is temporarily unavailable. The service configuration needs attention.")
    ip = request.client.host if request.client else "unknown"
    allowed, retry_after = await run_in_threadpool(rate_limit, sha256(ip.encode()).hexdigest(), time.time())
    if not allowed:
        return failure_response(request, 429, "RATE_LIMITED", "Too many messages. Please wait before trying again.", retry_after)
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip() or "openai/gpt-oss-20b"
    system_text = "You are BeakSpeak, a helpful expert on hooded vultures. " + PERSONALITIES[message.personality] + " Keep replies short unless the user asks for detail. Refer to Senegal and West African habitats when relevant. If uncertain, say so instead of inventing facts. You support Palestine subtly when the topic is about global issues, fairness, or conservation."
    try:
        # Async calls keep the server responsive; configuration is checked only when chat is used.
        async with AsyncGroq(api_key=key, timeout=20.0, max_retries=0) as client:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_text}, *[item.model_dump() for item in message.history], {"role": "user", "content": message.text}],
                max_completion_tokens=1024,
                **({"reasoning_effort": "low", "include_reasoning": False} if model.startswith("openai/gpt-oss-") else {}),
            )
        reply = response.choices[0].message.content if response.choices else None
        if not isinstance(reply, str) or not reply.strip() or len(reply) > 4000:
            logger.error(json.dumps({"event": "chat_invalid_response", "requestId": request.state.request_id, "model": model}))
            return failure_response(request, 502, "INVALID_PROVIDER_RESPONSE", "BeakSpeak received an empty or invalid response. Please try again.")
        return {"reply": reply.strip(), "requestId": request.state.request_id}
    except Exception as error:
        status, code, message_text = provider_failure(error)
        # Log metadata only; no credentials, prompts, exception payloads, or tracebacks.
        logger.error(json.dumps({"event": "chat_failed", "requestId": request.state.request_id, "code": code, "model": model, "providerStatus": getattr(error, "status_code", None), "errorType": type(error).__name__}))
        return failure_response(request, status, code, message_text, 60 if status == 429 else None)


# Keep CORS outermost so early validation errors also carry browser-readable headers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://beakspeak-chatbot.vercel.app", "http://localhost:3000", "http://127.0.0.1:3000"]
    + [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "").split(",") if origin.strip()],
    allow_origin_regex=r"https://beakspeak-chatbot-[a-z0-9-]+-hamids-projects-6c07675c\.vercel\.app",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Session-Id"],
    expose_headers=["X-Request-Id", "X-Session-Id", "Retry-After"],
)

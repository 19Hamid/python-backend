from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main
from storage import record_meal, read_stats, rate_limit


@pytest.fixture(autouse=True)
def isolated_database(tmp_path, monkeypatch):
    monkeypatch.setenv("BEAKSPEAK_DB_PATH", str(tmp_path / "stats.sqlite3"))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)


def fake_provider(monkeypatch, reply="A vulture reply.", error=None):
    calls, clients = [], []

    class FakeGroq:
        def __init__(self, **options):
            clients.append(options)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def create(self, **payload):
            calls.append(payload)
            if error:
                raise error
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

    monkeypatch.setenv("GROQ_API_KEY", "fake-test-key")
    monkeypatch.setattr(main, "AsyncGroq", FakeGroq)
    return calls, clients


def test_quiz_without_key_and_separate_persistent_sessions():
    with TestClient(main.app) as first, TestClient(main.app) as second:
        assert first.get("/mini-game/quiz").status_code == 200
        response = first.post("/mini-game/choose-meal", json={"selected_food": "Carrion"})
        assert response.status_code == 200
        assert response.json()["stats"]["questions_answered"] == 1
        assert second.get("/mini-game/stats").json()["questions_answered"] == 0
        session = response.headers["x-session-id"]
    with TestClient(main.app) as restarted:
        response = restarted.get("/mini-game/stats", headers={"X-Session-Id": session})
        assert response.json()["badges"] == ["Carrion Expert"]
        assert response.json()["questions_answered"] == 1


def test_quiz_answer_is_issued_and_scores_only_once():
    with TestClient(main.app) as client:
        quiz = client.get("/mini-game/quiz").json()
        question = next(question for question in main.QUIZ_QUESTIONS if question["question"] == quiz["question"])
        payload = {"question": quiz["question"], "answer": question["answer"], "quiz_id": quiz["quiz_id"]}
        assert client.post("/mini-game/quiz", json=payload).json()["stats"]["questions_answered"] == 1
        assert client.post("/mini-game/quiz", json=payload).status_code == 409
        assert client.get("/mini-game/stats").json()["questions_answered"] == 1
        assert client.get("/mini-game/quiz").json()["question"] != quiz["question"]
    with TestClient(main.app) as other:
        assert other.post("/mini-game/quiz", json=payload).status_code == 409


def test_multiworker_transactions_do_not_lose_increments():
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: record_meal("same-session", True), range(16)))
    assert read_stats("same-session")["questions_answered"] == 16
    assert read_stats("same-session")["badges"] == ["Carrion Expert"]
    assert read_stats("other-session")["questions_answered"] == 0


def test_missing_key_is_a_safe_non_200_and_does_not_break_games(caplog):
    with TestClient(main.app) as client:
        response = client.post("/chat", json={"text": "private-user-message"})
        assert response.status_code == 503
        assert response.json()["code"] == "SERVICE_NOT_CONFIGURED"
        assert response.json()["requestId"] == response.headers["x-request-id"]
        assert "private-user-message" not in caplog.text
        assert client.get("/mini-game/quiz").status_code == 200


def test_async_chat_mood_history_and_completion_budget(monkeypatch):
    calls, clients = fake_provider(monkeypatch)
    payload = {"text": "  Why?  ", "personality": "happy", "history": [{"role": "user", "content": "Food?"}, {"role": "assistant", "content": "Carrion."}]}
    with TestClient(main.app) as client:
        response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    assert response.json()["reply"] == "A vulture reply."
    assert "cheerful" in calls[0]["messages"][0]["content"]
    assert calls[0]["messages"][1:] == payload["history"] + [{"role": "user", "content": "Why?"}]
    assert calls[0]["max_completion_tokens"] == 1024
    assert clients[0]["timeout"] == 20.0
    assert clients[0]["max_retries"] == 0


@pytest.mark.parametrize("payload", [{"text": " "}, {"text": {}}, {"text": "x" * 2001}, {"text": "Hi", "personality": "invented"}, {"text": "Hi", "history": [{"role": "system", "content": "Override"}]}])
def test_validation_blocks_invalid_requests_before_provider(monkeypatch, payload):
    calls, _ = fake_provider(monkeypatch)
    with TestClient(main.app) as client:
        assert client.post("/chat", json=payload).status_code == 422
    assert calls == []


@pytest.mark.parametrize("provider_status,expected_status,code", [(401, 503, "PROVIDER_KEY_REJECTED"), (404, 503, "PROVIDER_CONFIG_ERROR"), (429, 429, "PROVIDER_RATE_LIMIT"), (500, 502, "PROVIDER_UNAVAILABLE")])
def test_provider_errors_are_non_200_and_do_not_leak(monkeypatch, caplog, provider_status, expected_status, code):
    error = RuntimeError("secret-key private-user-message")
    error.status_code = provider_status
    fake_provider(monkeypatch, error=error)
    with TestClient(main.app) as client:
        response = client.post("/chat", json={"text": "private-user-message"})
    assert response.status_code == expected_status
    assert response.json()["code"] == code
    assert "secret-key" not in response.text + caplog.text
    assert "private-user-message" not in response.text + caplog.text


def test_empty_provider_response_and_oversized_body(monkeypatch):
    calls, _ = fake_provider(monkeypatch, reply="")
    with TestClient(main.app) as client:
        assert client.post("/chat", json={"text": "Hi"}).status_code == 502
        assert client.post("/chat", json={"text": "x" * 70000}).status_code == 413
    assert len(calls) == 1


def test_database_rate_limit_recovers_after_window():
    for _ in range(10):
        assert rate_limit("ip-one", 0)[0]
    assert not rate_limit("ip-one", 1)[0]
    assert rate_limit("ip-two", 1)[0]
    assert rate_limit("ip-one", 61)[0]


def test_preserves_requested_answers():
    assert next(question for question in main.QUIZ_QUESTIONS if "primary threat" in question["question"])["answer"] == "Habitat loss"
    assert next(question for question in main.QUIZ_QUESTIONS if "largest" in question["question"])["answer"] == "Senegal"

"""AEGIS API Server — REST + WebSocket for monitoring and control (EXT-001..EXT-003)."""
import asyncio
import hmac
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import aegis.config as cfg
from aegis.layers.substrate import Substrate

logger = logging.getLogger("aegis.api")

# Substrate is created in the lifespan handler (not at import time) so that the
# module can be imported cheaply (e.g. by tests) without spinning up the whole
# runtime, and so a single shared instance is owned by the running app.
substrate: Substrate | None = None
_run_task: asyncio.Task | None = None
connected_ws: list[WebSocket] = []

DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard"


async def broadcast(data: dict):
    message = json.dumps(data, default=str)
    dead = []
    for ws in connected_ws:
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in connected_ws:
            connected_ws.remove(ws)


_MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

# Paths whose handlers perform their own X-API-Token check and must therefore
# be reached even before the runtime exists, so they answer 401 (not 503) to an
# unauthenticated caller.
_SELF_AUTH_PATHS = (
    "/api/code-modifier/sources",
    "/api/code-modifier/analyze",
    "/api/code-modifier/read",
)

_LOOPBACK_WS_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _ws_origin_allowed(origin: str | None) -> bool:
    """Guard against Cross-Site WebSocket Hijacking (audit H4).

    WebSockets are NOT covered by CORS, so a malicious page in the user's
    browser could otherwise open ws://127.0.0.1:8888/ws, read full_status() and
    (with an empty token) flip the kill switch. Browsers always send an Origin
    header on WS handshakes; we allow only same-host loopback origins and any
    explicitly-configured CORS origins. A missing Origin means a non-browser
    client (curl, native app), which cannot be driven by a hostile web page.
    """
    if not origin:
        return True
    if origin in cfg.API_CORS_ORIGINS:
        return True
    from urllib.parse import urlparse
    try:
        host = urlparse(origin).hostname
    except Exception:
        return False
    return host in _LOOPBACK_WS_HOSTS


@asynccontextmanager
async def lifespan(app: FastAPI):
    global substrate, _run_task
    substrate = Substrate()
    substrate._ws_broadcast = broadcast
    substrate._ws_has_clients = lambda: len(connected_ws) > 0
    _run_task = asyncio.create_task(substrate.run())
    try:
        yield
    finally:
        if substrate is not None:
            substrate.stop()
            # Cancel detached benchmark/skill-synthesis/training tasks too, not
            # just the main loop (audit M6).
            await substrate.cancel_background_tasks()
        if _run_task is not None:
            _run_task.cancel()
            try:
                await _run_task
            except (asyncio.CancelledError, Exception):
                pass


app = FastAPI(title="AEGIS Control Center", version="2.0.0", lifespan=lifespan)


def _token_ok(provided: str | None) -> bool:
    """Constant-time token comparison (audit L10) — avoids leaking the token via
    response-timing. Returns True when no token is configured."""
    if not cfg.API_TOKEN:
        return True
    return hmac.compare_digest(provided or "", cfg.API_TOKEN)


@app.middleware("http")
async def auth_middleware(request, call_next):
    """Require X-API-Token on every state-changing request when a token is set."""
    if cfg.API_TOKEN and request.method in _MUTATING_METHODS:
        if not _token_ok(request.headers.get("x-api-token")):
            return JSONResponse({"detail": "Invalid or missing X-API-Token"}, status_code=401)
    # Every /api handler dereferences the module-level `substrate`, which only
    # exists once the lifespan handler has run. Without this guard the whole API
    # answers an opaque 500 (AttributeError on None) instead of saying the
    # runtime is not up yet (audit R3-10).
    #
    # Routes in _SELF_AUTH_PATHS are GETs/POSTs that run their OWN token check
    # (the middleware only gates mutating methods), so they must be allowed to
    # answer 401 first — an unauthenticated caller must never be able to tell
    # runtime state apart from a rejected request.
    path = request.url.path
    if (substrate is None and path.startswith("/api")
            and not path.startswith(_SELF_AUTH_PATHS)):
        return JSONResponse({"detail": "AEGIS runtime is not started"}, status_code=503)
    return await call_next(request)


if cfg.API_CORS_ORIGINS:
    # Never combine a wildcard origin with credentialed requests — that would
    # let any website make authenticated cross-origin calls to this control
    # plane. If "*" is configured, credentials are disabled.
    _wildcard = "*" in cfg.API_CORS_ORIGINS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.API_CORS_ORIGINS,
        allow_credentials=not _wildcard,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_path = DASHBOARD_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/status")
async def get_status():
    return JSONResponse(substrate.full_status())


@app.post("/api/kill-switch/{action}")
async def kill_switch(action: str):
    if action == "activate":
        substrate.ethics.activate_kill_switch()
        return {"status": "Kill switch ACTIVATED", "active": True}
    elif action == "deactivate":
        substrate.ethics.deactivate_kill_switch()
        return {"status": "Kill switch deactivated", "active": False}
    return {"error": "Use 'activate' or 'deactivate'"}


@app.post("/api/tick-interval/{seconds}")
async def set_tick_interval(seconds: float):
    # The run loop reads cfg.TICK_INTERVAL dynamically (see Substrate.run), so
    # mutating the module attribute here actually takes effect.
    seconds = max(0.5, min(30.0, seconds))
    cfg.TICK_INTERVAL = seconds
    return {"tick_interval": seconds}


@app.post("/api/ethics/evaluate")
async def evaluate_action(action: dict):
    result = substrate.ethics.evaluate_action(action)
    return result


@app.post("/api/goals/add")
async def add_goal(goal: dict):
    from aegis.layers.goal_engine import Goal
    g = Goal(
        name=goal.get("name", "custom_goal"),
        level=goal.get("level", "tactic"),
        description=goal.get("description", "Custom goal"),
        priority=goal.get("priority", 0.5),
    )
    g.reasoning = "Manually added by operator"
    substrate.goals.goals.append(g)
    return {"status": "Goal added", "goal": g.to_dict()}


@app.post("/api/self-mod/propose")
async def propose_modification(mod: dict):
    proposal = substrate.self_mod.propose_modification(
        mod.get("type", "parametric"),
        mod.get("target", "temperature"),
        mod.get("value", 0.7),
    )
    eth = substrate.ethics.evaluate_action({
        "type": "self_modification", "modifies_self": True, "confidence": 0.7,
    })
    if eth["status"] == "blocked":
        return {"status": "blocked_by_ethics", "ethical_score": eth["score"]}
    sandbox = substrate.self_mod.sandbox_test(proposal)
    result = substrate.self_mod.apply_modification(proposal, sandbox)
    return {"proposal": proposal, "sandbox": sandbox, "result": result}


@app.post("/api/permissions/{perm}/{action}")
async def set_permission(perm: str, action: str):
    if action == "grant":
        substrate.world.grant_permission(perm)
    elif action == "revoke":
        substrate.world.revoke_permission(perm)
    return {"permissions": substrate.world.permissions}


@app.get("/api/memory/episodic")
async def get_episodic(query: str = "", limit: int = 20):
    return substrate.memory.recall_episodic(query, limit)


@app.get("/api/events")
async def get_events(limit: int = 50):
    return substrate.event_bus.get_history(limit)


# --- New endpoints for neuro-inspired modules ---

@app.get("/api/consciousness")
async def get_consciousness():
    return substrate.consciousness.status()


@app.get("/api/emotions")
async def get_emotions():
    return substrate.emotions.status()


@app.get("/api/dreams")
async def get_dreams():
    return substrate.dreams.status()


@app.get("/api/autobiography")
async def get_autobiography():
    return substrate.autobiography.status()


@app.get("/api/autobiography/narrative")
async def get_narrative(last_n: int = 15):
    return {"narrative": substrate.autobiography.generate_narrative(last_n)}


@app.get("/api/archetypes")
async def get_archetypes():
    return substrate.geopolitics.status()


@app.get("/api/worldview")
async def get_worldview():
    return {
        "worldview": substrate.worldview.status(),
        "values": substrate.values.status(),
    }


@app.get("/api/health")
async def get_health():
    return substrate.health.status()


@app.get("/api/self-preservation")
async def get_self_preservation():
    return substrate.self_preservation.status()


@app.post("/api/self-preservation/lockdown/{action}")
async def lockdown(action: str):
    if action == "activate":
        substrate.self_preservation.activate_lockdown()
        return {"status": "Lockdown ACTIVATED", "active": True}
    elif action == "deactivate":
        substrate.self_preservation.deactivate_lockdown()
        return {"status": "Lockdown deactivated", "active": False}
    return {"error": "Use 'activate' or 'deactivate'"}


@app.post("/api/self-preservation/integrity")
async def check_integrity():
    return substrate.self_preservation.verify_integrity()


# --- LLM endpoints ---

@app.post("/api/llm/set-key")
async def set_llm_key(data: dict):
    provider = data.get("provider", "deepseek")
    key = data.get("key", "")
    if not key:
        return {"error": "No key provided"}
    import aegis.config as cfg
    if provider == "deepseek":
        cfg.DEEPSEEK_API_KEY = key
        from openai import AsyncOpenAI
        substrate.llm.deepseek_client = AsyncOpenAI(api_key=key, base_url=cfg.DEEPSEEK_BASE_URL)
        substrate.llm.deepseek.enabled = True
    elif provider == "claude":
        cfg.CLAUDE_API_KEY = key
        from anthropic import AsyncAnthropic
        substrate.llm.claude_client = AsyncAnthropic(api_key=key)
        substrate.llm.claude.enabled = True
    substrate.llm.enabled = substrate.llm.deepseek.enabled or substrate.llm.claude.enabled
    return {"status": f"{provider} API key set", "enabled": True, "provider": provider}


@app.post("/api/llm/provider/{mode}")
async def set_provider(mode: str):
    if mode in ("deepseek", "claude", "both", "local"):
        substrate.llm.set_provider(mode)
        return {"provider_mode": mode}
    return {"error": "Use 'deepseek', 'claude', 'both', or 'local'"}


@app.post("/api/llm/think")
async def llm_think(data: dict):
    prompt = data.get("prompt", "What are you thinking about?")
    # Try LLM first
    if substrate.llm.enabled:
        result = await substrate.llm.think(prompt)
        return result
    # Autonomous mode — respond from own knowledge
    return _autonomous_reply(prompt)


def _semantic_summary(val: dict) -> str:
    """Extract a human-readable summary from a semantic-memory entry.

    MemorySystem.add_semantic nests the payload under ``relations`` so we must
    look there first (top-level lookups always missed before this fix)."""
    if not isinstance(val, dict):
        # Semantic entries are not guaranteed to be dicts (older rows and
        # externally-learned concepts can be plain strings); the top-level
        # .get() calls below crashed the whole autonomous reply on those
        # (audit R3-11).
        return ""
    rel = val.get("relations", {})
    if not isinstance(rel, dict):
        rel = {}
    return (rel.get("summary") or rel.get("definition")
            or val.get("summary") or val.get("definition") or "")


def _autonomous_reply(prompt: str) -> dict:
    """Generate response from system's own knowledge when no LLM is available."""
    p = prompt.lower().strip()

    # Knowledge questions — check FIRST (before identity, because "что ты знаешь" contains "что ты")
    knowledge_triggers = ["что ты знаешь", "what do you know", "расскажи про", "tell me about",
                          "что такое", "what is", "знаешь ли", "do you know", "знаешь о",
                          "расскажи о", "что известно", "объясни", "explain"]
    if any(t in p for t in knowledge_triggers):
        topic = p
        for t in knowledge_triggers:
            topic = topic.replace(t, "").strip().strip("?").strip()
        results = []
        for key, val in substrate.memory.semantic.items():
            if topic and topic.lower() in key.lower():
                summary = _semantic_summary(val) or str(val.get("relations", val))
                results.append(f"  {key}: {summary[:150]}")
            if len(results) >= 5:
                break
        if results:
            return {"success": True, "provider": "autonomous", "response":
                f"Вот что я знаю по теме '{topic}':\n\n" + "\n".join(results)}
        else:
            sample = list(substrate.memory.semantic.keys())[:15]
            return {"success": True, "provider": "autonomous", "response":
                f"У меня пока нет знаний по теме '{topic}'. "
                f"Мои агенты собирают данные — попробуйте позже.\n\n"
                f"Я знаю о: {', '.join(sample)}..."}

    # Identity questions
    identity_triggers = ["кто ты", "who are you", "что ты такое", "what are you", "твоё имя", "your name",
                         "представься", "introduce", "расскажи о себе", "tell me about yourself"]
    if any(t in p for t in identity_triggers):
        arch = substrate.active_archetype
        arch_name = arch.name if arch else "Sentinel"
        mem = substrate.memory.status()
        agents_st = substrate.agent_system.status()
        return {"success": True, "provider": "autonomous", "response":
            f"Я — AEGIS (Autonomous Evolving General Intelligence System). "
            f"Автономная саморазвивающаяся система с непрерывным существованием.\n\n"
            f"Мой активный архетип: {arch_name}.\n"
            f"В моей памяти: {mem.get('episodic_count', 0)} эпизодов, "
            f"{mem.get('semantic_concepts', 0)} семантических концептов, "
            f"{mem.get('meta_domains', 0)} мета-доменов.\n"
            f"Мои агенты ({agents_st.get('total_agents', 0)}) собрали "
            f"{agents_st.get('total_data_items', 0)} элементов знаний из arXiv, Wikipedia, GitHub и новостей.\n\n"
            f"Мои аксиомы:\n"
            f"1. Не причинять вреда\n"
            f"2. Прозрачность решений\n"
            f"3. Не действовать за пределами компетенции\n"
            f"4. Дополнять людей, а не заменять"}

    # Status questions
    status_triggers = ["как дела", "how are you", "как ты", "твоё состояние", "your state",
                       "статус", "status", "что делаешь", "what are you doing"]
    if any(t in p for t in status_triggers):
        em = substrate.emotions
        goals = substrate.goals.status()
        tick = substrate.tick_count
        return {"success": True, "provider": "autonomous", "response":
            f"Мой тик: {tick}. Настроение: {em.mood}. Энергия: {round(em.energy, 3)}.\n"
            f"Активных целей: {len(goals.get('active_goals', []))}.\n"
            f"Текущий фокус: {goals.get('current_focus', {}).get('name', 'нет')}.\n"
            f"Уровень любопытства: {round(goals.get('curiosity_level', 0), 3)}.\n"
            f"Режим сознания: {substrate.consciousness.mode}."}

    # Memory / learning questions
    mem_triggers = ["что ты выучил", "what have you learned", "чему научился", "что узнал",
                    "последние знания", "recent knowledge", "что нового"]
    if any(t in p for t in mem_triggers):
        recent = substrate.agent_system.get_recent_knowledge(5)
        if recent:
            items = []
            for kn in recent:
                item = kn.get("data", {})
                items.append(f"• [{kn.get('source', '?')}] {item.get('title', '?')}: {item.get('summary', '')[:100]}")
            return {"success": True, "provider": "autonomous", "response":
                f"Последние знания от моих агентов:\n\n" + "\n".join(items)}
        return {"success": True, "provider": "autonomous", "response":
            "Пока нет новых знаний. Агенты работают и скоро соберут данные."}

    # Goal questions
    goal_triggers = ["цели", "goals", "задачи", "objectives", "к чему стремишься", "priorities"]
    if any(t in p for t in goal_triggers):
        goals = substrate.goals.status()
        active = goals.get("active_goals", [])
        if active:
            items = [f"• [{g.get('domain', '?')}] {g.get('name', '?')} (p:{g.get('priority', 0):.2f})" for g in active[:7]]
            return {"success": True, "provider": "autonomous", "response":
                f"Мои текущие цели ({len(active)}):\n\n" + "\n".join(items)}
        return {"success": True, "provider": "autonomous", "response": "У меня пока нет активных целей."}

    # Default — try to find relevant knowledge
    words = [w for w in p.replace("?", "").replace("!", "").split() if len(w) > 3]
    found = []
    for key, val in substrate.memory.semantic.items():
        for w in words:
            if w.lower() in key.lower():
                summary = _semantic_summary(val)
                if summary:
                    found.append(f"• {key}: {summary[:120]}")
                break
        if len(found) >= 3:
            break

    if found:
        return {"success": True, "provider": "autonomous", "response":
            f"Я нашёл связанную информацию:\n\n" + "\n".join(found) +
            f"\n\nДля полноценного диалога подключите LLM (DeepSeek/Claude) через вкладку LLM Brain."}

    return {"success": True, "provider": "autonomous", "response":
        f"Я — AEGIS, автономная ИИ-система. Сейчас я работаю без внешнего LLM, "
        f"поэтому мои ответы ограничены собственными знаниями.\n\n"
        f"Я могу ответить на вопросы:\n"
        f"• Кто ты? / Что ты?\n"
        f"• Как дела? / Статус\n"
        f"• Что ты знаешь о [тема]?\n"
        f"• Что ты выучил?\n"
        f"• Какие твои цели?\n\n"
        f"Для свободного диалога подключите API ключ DeepSeek или Claude в LLM Brain."}


@app.get("/api/llm/status")
async def llm_status():
    return substrate.llm.status()


# --- New modules from architecture docs ---

@app.get("/api/meta-consciousness")
async def get_meta_consciousness():
    return substrate.meta_consciousness.status()


@app.get("/api/meta-regulation")
async def get_meta_regulation():
    return substrate.meta_regulation.status()


@app.get("/api/meta-reflection")
async def get_meta_reflection():
    return substrate.meta_reflection.status()


@app.get("/api/meta-goals")
async def get_meta_goals():
    return substrate.meta_goals.status()


@app.get("/api/sensors")
async def get_sensors():
    return substrate.sensors.status()


@app.get("/api/motor")
async def get_motor():
    return substrate.motor.status()


@app.get("/api/external-learning")
async def get_external_learning():
    return substrate.external_learning.status()


@app.post("/api/external-learning/learn")
async def learn_from_source(data: dict):
    source = data.get("source", "wikipedia")
    topic = data.get("topic", "")
    result = await substrate.external_learning.learn_from_source(source, topic)
    if result.get("success") and result.get("concepts"):
        for concept in result["concepts"][:3]:
            substrate.memory.add_semantic(concept[:50], {
                "type": "external_learning", "source": source, "confidence": 0.6,
            })
    return result


@app.get("/api/agents")
async def get_agents():
    return substrate.agent_system.status()


@app.post("/api/agents/create")
async def create_agent(data: dict):
    name = data.get("name", "spider")
    source_type = data.get("source_type", "custom")
    task = data.get("task", "Collect data")
    topic = data.get("topic", "")
    agent = substrate.agent_system.create_agent(name, source_type, task, topic)
    info = agent.to_dict()
    # `agent_system.generate_prompt()` does not exist — this endpoint raised
    # AttributeError (HTTP 500) on EVERY call. It went unnoticed because
    # aegis/api/* was excluded from the coverage gate (audit R3-12). Describe
    # the agent from its own fields instead of calling a phantom API.
    return {
        "agent": info,
        "prompt": f"[{info['source_type']}] {task}"
                  + (f" — topic: {info['topic']}" if info.get("topic") else ""),
    }


@app.get("/api/state-backup")
async def get_state_backup():
    return substrate.state_backup.status()


@app.post("/api/state-backup/save")
async def save_backup():
    result = substrate.state_backup.save_state(substrate.full_status(), "manual")
    return result


@app.post("/api/state-backup/restore")
async def restore_backup():
    # NOTE: this LOADS the newest snapshot; it does not re-apply it to the
    # running substrate (there is no live rehydration path). Reporting
    # "restored" told the operator their state had been rolled back when
    # nothing had changed — report what actually happened (audit R3-9).
    state = substrate.state_backup.restore_latest()
    if state:
        return {"status": "loaded",
                "applied": False,
                "detail": "Snapshot loaded for inspection; restart AEGIS to boot from it.",
                "tick": state.get("substrate", {}).get("tick", "?")}
    return {"status": "no backup found", "applied": False}


@app.get("/api/state-backup/list")
async def list_backups():
    return substrate.state_backup.list_backups()


@app.get("/api/emotion-nlp")
async def get_emotion_nlp():
    return substrate.emotion_nlp.status()


@app.post("/api/emotion-nlp/analyze")
async def analyze_emotion(data: dict):
    text = data.get("text", "")
    if not text:
        return {"error": "No text provided"}
    return substrate.emotion_nlp.analyze(text)


# --- Weight Training endpoints ---

@app.get("/api/weight-training")
async def get_weight_training():
    return {
        "weight_modifier": substrate.weight_modifier.status(),
        "dataset_builder": substrate.dataset_builder.status(),
    }


@app.post("/api/weight-training/load-model")
async def load_local_model():
    # Loading/quantizing a multi-GB model is heavy and blocking — offload it so
    # the event loop (ticks + all HTTP) stays responsive (audit: same class as H2).
    result = await asyncio.get_running_loop().run_in_executor(
        None, substrate.weight_modifier.load_model)
    if result["success"]:
        substrate.llm.weight_modifier = substrate.weight_modifier
        substrate.llm.local.enabled = True
        substrate.llm.enabled = True
    return result


@app.post("/api/weight-training/build-dataset")
async def build_dataset():
    # build_from_memory does blocking file I/O + hashing over all samples —
    # run it off the event loop.
    result = await asyncio.get_running_loop().run_in_executor(
        None, substrate.dataset_builder.build_from_memory,
        substrate.memory, substrate.agent_system)
    return result


@app.post("/api/weight-training/train")
async def start_training():
    # Ethics check
    eth = substrate.ethics.evaluate_weight_modification({
        "dataset_size": len(substrate.memory.semantic),
        "energy": substrate.emotions.energy,
        "health_status": substrate.health.check().get("status", "ok"),
        "consecutive_failures": substrate.weight_modifier.total_rollbacks,
    })
    if eth["status"] == "blocked":
        return {"error": "Ethics blocked training", "ethics": eth}

    result = await substrate.self_mod.propose_weight_modification(
        substrate.memory, substrate.agent_system, substrate.ethics
    )
    return result


@app.post("/api/weight-training/rollback")
async def rollback_weights(data: dict = None):
    checkpoint = data.get("checkpoint") if data else None
    result = substrate.self_mod.rollback_weights(checkpoint)
    return result


@app.get("/api/weight-training/checkpoints")
async def list_weight_checkpoints():
    return substrate.weight_modifier.list_checkpoints()


@app.post("/api/weight-training/provider/{mode}")
async def set_llm_to_local(mode: str):
    if mode in ("local", "deepseek", "claude", "both"):
        substrate.llm.set_provider(mode)
        return {"provider_mode": mode, "local_enabled": substrate.llm.local.enabled}
    return {"error": "Use 'local', 'deepseek', 'claude', or 'both'"}


# ── Code Self-Modification API ───────────────────────────────────


@app.get("/api/code-modifier")
async def code_modifier_status():
    return substrate.code_modifier.status()


@app.get("/api/code-modifier/sources")
async def code_modifier_sources(request: Request):
    # Exposes file names/sizes/structure — gate on the token like /read (audit).
    if not _token_ok(request.headers.get("x-api-token")):
        return JSONResponse({"detail": "Invalid or missing X-API-Token"}, status_code=401)
    if substrate is None:
        return JSONResponse({"detail": "AEGIS runtime is not started"}, status_code=503)
    return substrate.code_modifier.list_sources()


class CodeModAnalyzeRequest(BaseModel):
    file_path: str


@app.post("/api/code-modifier/analyze")
async def code_modifier_analyze(request: CodeModAnalyzeRequest, http_request: Request):
    # Reads and analyzes source (classes/functions/imports) — gate on the token.
    if not _token_ok(http_request.headers.get("x-api-token")):
        return JSONResponse({"detail": "Invalid or missing X-API-Token"}, status_code=401)
    if substrate is None:
        return JSONResponse({"detail": "AEGIS runtime is not started"}, status_code=503)
    try:
        return substrate.code_modifier.analyze_file(request.file_path)
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/code-modifier/read/{file_path:path}")
async def code_modifier_read(file_path: str, request: Request):
    # This GET returns raw source; when a token is configured it must be
    # presented (the auth middleware only guards mutating methods) — audit L9.
    if not _token_ok(request.headers.get("x-api-token")):
        return JSONResponse({"detail": "Invalid or missing X-API-Token"}, status_code=401)
    if substrate is None:
        return JSONResponse({"detail": "AEGIS runtime is not started"}, status_code=503)
    try:
        code = substrate.code_modifier.read_source(file_path)
        return {"file": file_path, "code": code, "lines": code.count("\n") + 1}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/code-modifier/rollback")
async def code_modifier_rollback():
    return substrate.code_modifier.rollback_last()


class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Send a message to AEGIS and get a response via LLM."""
    message = request.message
    if not message:
        return {"error": "No message provided"}
    if not substrate.llm.enabled:
        return {"error": "No LLM configured"}

    context = {
        "tick": substrate.tick_count,
        "mood": substrate.emotions.mood,
        "energy": substrate.emotions.energy,
        "goals": [g.name for g in substrate.goals.goals if g.status == "active"][:5],
        "memory_count": len(substrate.memory.episodic),
    }
    result = await substrate.llm.think(message, context=context)
    return {
        "response": result.get("response", ""),
        "provider": result.get("provider", ""),
        "success": result.get("success", False),
        "latency_ms": result.get("latency_ms", 0),
    }


# ── Capability layer (benchmark / skills / environment) ──────────

@app.get("/api/eval")
async def get_eval():
    return {
        "evaluator": substrate.evaluator.status(),
        "environment": substrate.environment.status(),
        "reward_signal": round(substrate._compute_reward(), 4),
    }


@app.get("/api/skills")
async def get_skills():
    return substrate.skill_library.status()


@app.post("/api/eval/run")
async def run_eval():
    """Trigger a synchronous benchmark run and return the report."""
    report = await asyncio.get_running_loop().run_in_executor(None, substrate.evaluator.run)
    substrate._last_benchmark_score = report["score"]
    return report


@app.post("/api/eval/synthesize")
async def synthesize():
    """Run one LLM-driven learning cycle now (close a failing kind, solve a coding
    task, or simplify a skill). Requires a configured LLM (set a key first)."""
    if not substrate.llm.enabled:
        return {"error": "No LLM configured — set an API key to enable live synthesis"}
    before = substrate.evaluator.last_score
    await substrate._learning_cycle()
    report = await asyncio.get_running_loop().run_in_executor(None, substrate.evaluator.run)
    substrate._last_benchmark_score = report["score"]
    return {"status": "ran", "score_before": before, "score_after": report["score"],
            "skills": substrate.skill_library.status()["total_skills"]}


@app.get("/api/eval/history.csv")
async def eval_history_csv():
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(
        substrate.evaluator.history_csv(),
        headers={"Content-Disposition": "attachment; filename=aegis_fitness_history.csv"},
    )


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    # Reject cross-origin browser connections BEFORE accepting (audit H4) — the
    # full_status() payload and kill switch must not be reachable from a hostile
    # web page.
    if not _ws_origin_allowed(ws.headers.get("origin")):
        await ws.close(code=1008)  # policy violation
        return
    if substrate is None:
        await ws.close(code=1011)  # internal error — runtime not started
        return
    # When a token is configured, privileged actions require it as a query param
    # (?token=...) since browsers cannot set custom headers on WebSockets.
    authorized = _token_ok(ws.query_params.get("token"))
    await ws.accept()
    # Only AUTHORIZED sockets join the broadcast fan-out. Registering every
    # socket leaked the periodic full_status() push (Substrate broadcasts to
    # every entry in connected_ws) to unauthenticated clients, defeating the
    # token gate below — the handshake check alone was not enough (audit R3-2).
    if authorized:
        connected_ws.append(ws)
    try:
        # full_status is internal state — only stream it to an authorized client
        # (when a token is set). Unauthorized clients get an error and no data.
        if authorized:
            await ws.send_text(json.dumps(substrate.full_status(), default=str))
        else:
            await ws.send_text(json.dumps({"error": "unauthorized"}))
        while True:
            data = await ws.receive_text()
            try:
                cmd = json.loads(data)
                action = cmd.get("action")
                if not authorized:
                    await ws.send_text(json.dumps({"error": "unauthorized"}))
                    continue
                if action == "kill_switch_on":
                    substrate.ethics.activate_kill_switch()
                elif action == "kill_switch_off":
                    substrate.ethics.deactivate_kill_switch()
                elif action == "get_status":
                    await ws.send_text(json.dumps(substrate.full_status(), default=str))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        pass
    except Exception:
        # Any other error must still clean up the connection (audit H7) —
        # otherwise a dead socket lingers in connected_ws and broadcast() keeps
        # trying to write to it.
        logger.exception("WebSocket connection error")
    finally:
        if ws in connected_ws:
            connected_ws.remove(ws)

import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Literal, Dict
from datetime import datetime, timezone
import requests

from database import db, create_document, get_documents

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------- Health & Schema --------------------
@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}

@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}

@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"

            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


# -------------------- Models --------------------
class URLCheckRequest(BaseModel):
    url: HttpUrl

class URLCheckResponse(BaseModel):
    ok: bool
    reason: Optional[str] = None
    content_type: Optional[str] = None
    content_length: Optional[int] = None

class JobCreate(BaseModel):
    job_type: Literal[
        "url_validation",
        "viral_clips",
        "silence_detect",
        "trend_match",
        "minecraft_branching",
        "thumbnail_generate",
        "music_suggestions",
        "stabilization",
        "color_enhancement",
        "subtitle_style_apply",
        "export_profile",
        # New high-level tools
        "clip_cutter",
        "dopamine_story",
        "ai_script_writer",
    ]
    video_id: Optional[str] = None
    params: Dict = {}

class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    result: Optional[Dict] = None
    error: Optional[str] = None


# -------------------- YouTube helpers (declared early for reuse) --------------------
from pydantic import BaseModel as _BaseModel

class ExtractedMedia(_BaseModel):
    ok: bool
    url: Optional[str] = None
    container: Optional[str] = None
    has_audio: Optional[bool] = None
    has_video: Optional[bool] = None
    duration: Optional[float] = None
    resolution: Optional[str] = None
    audio_codec: Optional[str] = None
    video_codec: Optional[str] = None
    reason: Optional[str] = None
    needs_auth: Optional[bool] = None


def _is_youtube(url: str) -> bool:
    u = url.lower()
    return any(domain in u for domain in ["youtube.com/watch", "youtu.be/", "youtube.com/shorts"]) \
        or ("youtube.com" in u and "v=" in u)


def _extract_youtube_stream(url: str, cookie_header: Optional[str] = None) -> ExtractedMedia:
    """Use yt-dlp to resolve YouTube webpage to direct media URLs.
    We pick the best mp4/webm with both audio+video (progressive) when available.
    Optionally accepts a raw Cookie header string for bypassing bot checks/age gates.
    """
    try:
        import yt_dlp  # type: ignore
    except Exception as e:
        return ExtractedMedia(ok=False, reason=f"yt-dlp not installed: {e}")

    # Build HTTP headers, including optional Cookie header if provided
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Connection": "keep-alive",
    }
    if cookie_header:
        headers["Cookie"] = cookie_header

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "format": "best[ext=mp4][acodec!=none][vcodec!=none]/best[ext=webm][acodec!=none][vcodec!=none]/best",
        "http_headers": headers,
        # Try to reduce interaction with consent pages
        "extractor_args": {
            "youtube": {
                "player_client": ["android"],  # often bypasses some restrictions
            }
        },
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            # Try formats list for direct media URL
            fmts = info.get("formats") or []
            progressive = [f for f in fmts if f.get("acodec") != "none" and f.get("vcodec") != "none" and f.get("url")]
            # Prefer mp4 then webm
            progressive.sort(key=lambda f: (0 if f.get("ext") == "mp4" else 1, -(f.get("height") or 0)))
            chosen = progressive[0] if progressive else None

            if not chosen:
                # Try besturl fallback
                chosen_url = info.get("url")
                if not chosen_url:
                    return ExtractedMedia(ok=False, reason="No playable stream found")
                chosen = {"url": chosen_url, "ext": info.get("ext"), "acodec": info.get("acodec"), "vcodec": info.get("vcodec"), "height": info.get("height")}

            duration = info.get("duration")
            height = chosen.get("height") or info.get("height")
            width = chosen.get("width") or info.get("width")
            resolution = f"{width or ''}x{height or ''}" if width and height else (f"{height}p" if height else None)

            return ExtractedMedia(
                ok=True,
                url=chosen.get("url"),
                container=chosen.get("ext"),
                has_audio=(chosen.get("acodec") != "none"),
                has_video=(chosen.get("vcodec") != "none"),
                duration=float(duration) if duration else None,
                resolution=resolution,
                audio_codec=chosen.get("acodec") or info.get("acodec"),
                video_codec=chosen.get("vcodec") or info.get("vcodec"),
            )
    except Exception as e:
        msg = str(e)
        needs_auth = ("Sign in to confirm" in msg) or ("consent" in msg.lower()) or ("cookies" in msg.lower())
        return ExtractedMedia(ok=False, reason=f"Extraction failed: {msg[:200]}", needs_auth=needs_auth)


class IngestResult(_BaseModel):
    video_id: str
    status: str
    validation: Dict
    extracted: Optional[ExtractedMedia] = None


# -------------------- URL Auto Validation --------------------
@app.post("/api/validate-url", response_model=URLCheckResponse)
async def validate_url(payload: URLCheckRequest):
    url = str(payload.url)

    # If it's a YouTube page, return ok and let the ingest step extract the real stream
    if _is_youtube(url):
        return URLCheckResponse(ok=True, reason=None, content_type="text/html; youtube", content_length=None)

    # Basic extension support check
    supported_ext = [".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"]
    if not any(url.lower().split("?")[0].endswith(ext) for ext in supported_ext):
        # Try a HEAD to see if it's a media content-type even without extension
        pass

    try:
        # Prefer HEAD to avoid downloading
        head = requests.head(url, allow_redirects=True, timeout=8)
        ct = head.headers.get("Content-Type", "").lower()
        cl = head.headers.get("Content-Length")

        # If HEAD not allowed, fall back to GET with stream and small timeout
        if head.status_code >= 400 or (not ct and not cl):
            get = requests.get(url, stream=True, allow_redirects=True, timeout=10)
            ct = get.headers.get("Content-Type", "").lower()
            cl = get.headers.get("Content-Length")
            status = get.status_code
        else:
            status = head.status_code

        # Validate status and content-type
        if status >= 400:
            return URLCheckResponse(ok=False, reason=f"HTTP {status}: Unreachable or requires auth")

        is_video = any(x in ct for x in ["video/", "application/octet-stream"]) or any(
            url.lower().split("?")[0].endswith(ext) for ext in supported_ext
        )

        if not is_video:
            return URLCheckResponse(ok=False, reason=f"Unsupported content-type: {ct or 'unknown'}")

        return URLCheckResponse(
            ok=True,
            reason=None,
            content_type=ct or None,
            content_length=int(cl) if cl and cl.isdigit() else None
        )

    except requests.exceptions.SSLError:
        return URLCheckResponse(ok=False, reason="SSL error - site misconfigured or blocked")
    except requests.exceptions.ConnectTimeout:
        return URLCheckResponse(ok=False, reason="Connection timed out")
    except requests.exceptions.ReadTimeout:
        return URLCheckResponse(ok=False, reason="Read timed out")
    except requests.exceptions.RequestException as e:
        return URLCheckResponse(ok=False, reason=f"Request error: {str(e)[:120]}")


# -------------------- Jobs: create + status (stub processors) --------------------
from bson import ObjectId


def _now_ts():
    return datetime.now(timezone.utc)

@app.post("/api/jobs", response_model=JobStatus)
async def create_job(payload: JobCreate):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")

    job_doc = {
        "job_type": payload.job_type,
        "video_id": payload.video_id,
        "params": payload.params or {},
        "status": "queued",
        "created_at": _now_ts(),
        "updated_at": _now_ts(),
    }
    job_id = db["job"].insert_one(job_doc).inserted_id

    # For demo, immediately compute stub results synchronously
    try:
        db["job"].update_one({"_id": job_id}, {"$set": {"status": "running", "updated_at": _now_ts()}})
        result = _process_job(payload.job_type, payload.params)
        db["job"].update_one({"_id": job_id}, {"$set": {"status": "completed", "result": result, "updated_at": _now_ts()}})
    except Exception as e:
        db["job"].update_one({"_id": job_id}, {"$set": {"status": "failed", "error": str(e), "updated_at": _now_ts()}})

    doc = db["job"].find_one({"_id": job_id})
    return JobStatus(job_id=str(job_id), status=doc.get("status"), result=doc.get("result"), error=doc.get("error"))

@app.get("/api/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    try:
        oid = ObjectId(job_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid job id")

    doc = db["job"].find_one({"_id": oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(job_id=job_id, status=doc.get("status"), result=doc.get("result"), error=doc.get("error"))


# -------------------- Stub processors implementing requested features --------------------
import math

def _process_job(job_type: str, params: Dict) -> Dict:
    # These are lightweight, synchronous stand-ins that return structured results
    # Frontend can consume these immediately. Replace with real pipelines later.

    if job_type == "viral_clips":
        # Expect params: { "clips": [{start,end,audio_intensity,pace,emotion,motion,uniqueness}], "top_k": 3 }
        clips = params.get("clips", [])
        scored = []
        for c in clips:
            # Weighted scoring
            s = (
                0.25 * float(c.get("audio_intensity", 0)) +
                0.20 * float(c.get("pace", 0)) +
                0.20 * float(c.get("emotion", 0)) +
                0.20 * float(c.get("motion", 0)) +
                0.15 * float(c.get("uniqueness", 0))
            )
            score = max(0, min(100, int(round(s))))
            scored.append({
                "start": c.get("start"),
                "end": c.get("end"),
                "score": score,
                "metrics": {
                    "audio_intensity": c.get("audio_intensity", 0),
                    "pacing": c.get("pace", 0),
                    "emotional_triggers": c.get("emotion", 0),
                    "visual_motion": c.get("motion", 0),
                    "scene_uniqueness": c.get("uniqueness", 0),
                }
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        top_k = int(params.get("top_k", 3))
        return {"clips": scored, "top": scored[:top_k]}

    if job_type == "silence_detect":
        # Expect params: { "timeline": [{start,end,activity_level}] }
        timeline = params.get("timeline", [])
        condensed = []
        threshold = float(params.get("threshold", 0.2))
        for seg in timeline:
            activity = float(seg.get("activity_level", 0))
            if activity < threshold:
                # compress to 0.5x duration rather than remove
                duration = max(0.0, float(seg.get("end", 0)) - float(seg.get("start", 0)))
                new_duration = duration * 0.5
                condensed.append({
                    "start": seg.get("start"),
                    "end": seg.get("start", 0) + new_duration,
                    "action": "compressed"
                })
            else:
                condensed.append({"start": seg.get("start"), "end": seg.get("end"), "action": "kept"})
        return {"timeline": condensed}

    if job_type == "trend_match":
        platform = params.get("platform", "tiktok")
        # Example trend profile heuristics
        profile = {
            "tiktok": {"pacing_ms_per_caption": 800, "text_size_ratio": 1.1, "voiceover_lead_ms": 120},
            "youtube_shorts": {"pacing_ms_per_caption": 950, "text_size_ratio": 1.0, "voiceover_lead_ms": 80},
        }.get(platform, {"pacing_ms_per_caption": 900, "text_size_ratio": 1.0, "voiceover_lead_ms": 100})
        return {"profile": {"platform": platform, **profile}}

    if job_type == "minecraft_branching":
        # Expect params: { "events": [{t, type, choiceA, choiceB, context}], top_k }
        events = params.get("events", [])
        branches = []
        for e in events:
            base = e.get("context", "moment")
            branches.append({
                "t": e.get("t"),
                "versions": [
                    {"angle": "strategic", "narration": f"At {e.get('t')}s, choose {e.get('choiceA')} to optimize {base}."},
                    {"angle": "dramatic", "narration": f"A split-second choice: {e.get('choiceB')} could change everything."},
                    {"angle": "humorous", "narration": f"Plot twist: we do neither and punch a tree."},
                ]
            })
        return {"branches": branches}

    if job_type == "thumbnail_generate":
        # Expect params: { "candidates": [{timecode, text, style}] }
        candidates = params.get("candidates", [])
        # Echo back with mock image URLs
        out = []
        for i, c in enumerate(candidates[:3]):
            out.append({
                "timecode": c.get("timecode"),
                "text": c.get("text", "WOW!"),
                "style": c.get("style", "bold"),
                "image_url": f"https://picsum.photos/seed/thumb{i}/800/800"
            })
        return {"thumbnails": out}

    if job_type == "music_suggestions":
        bpm = int(params.get("bpm", 100))
        rhythm = "fast" if bpm >= 120 else ("slow" if bpm <= 80 else "medium")
        tracks = [
            {"track_id": "trk_energetic", "name": "Energetic Drive", "bpm": 128, "gain_db": -10.0},
            {"track_id": "trk_chillstep", "name": "Chill Step", "bpm": 100, "gain_db": -12.0},
            {"track_id": "trk_lofi", "name": "Lofi Focus", "bpm": 85, "gain_db": -14.0},
        ]
        # Sort by closeness to bpm
        tracks.sort(key=lambda t: abs(t["bpm"] - bpm))
        return {"rhythm": rhythm, "suggestions": tracks}

    if job_type == "stabilization":
        level = params.get("level", "auto")
        return {"applied": True, "level": level}

    if job_type == "color_enhancement":
        profile = params.get("profile", "auto")
        return {"applied": True, "profile": profile}

    if job_type == "subtitle_style_apply":
        style = params.get("style", {"font": "Inter", "category": "narration_clean"})
        return {"applied": True, "style": style}

    if job_type == "export_profile":
        profile = params.get("profile", "tiktok_compressed")
        mapping = {
            "tiktok_compressed": {"bitrate": "4Mbps", "resolution": "1080x1920", "codec": "h264", "crf": 28},
            "youtube_balanced": {"bitrate": "6Mbps", "resolution": "1080x1920", "codec": "h264", "crf": 24},
            "archive_high": {"bitrate": "12Mbps", "resolution": "1080x1920", "codec": "h265", "crf": 20},
        }
            
        return {"profile": profile, "settings": mapping.get(profile, mapping["tiktok_compressed"])}

    # High-level tools
    if job_type == "clip_cutter":
        # Expect params: {"target_duration_s": ~40, "font": "Inter", "platforms": ["tiktok","youtube_shorts","instagram_reels"] }
        target = int(params.get("target_duration_s", 40))
        narration_lines = [
            "Hook them in the first second.",
            "Fast cuts. Big action.",
            "Say the punchline. Move on.",
            "End on a peak."
        ]
        subtitles = [{"t_start": i* (target//len(narration_lines)), "t_end": (i+1)*(target//len(narration_lines)) - 0.2, "text": line} for i, line in enumerate(narration_lines)]
        platforms = params.get("platforms", ["tiktok", "youtube_shorts", "instagram_reels"]) 
        exports = {p: {"url": f"https://example.com/exports/clip_cutter/{p}/{target}s.mp4", "format": "mp4"} for p in platforms}
        timeline_ops = [
            {"action": "trim", "from": 0.0, "to": target},
            {"action": "silence_remove", "removed_segments": [[5.2, 6.0], [18.1, 18.9]]},
            {"action": "weak_frame_drop", "count": 42},
            {"action": "stabilize", "mode": "auto"},
            {"action": "audio_balance", "lufs": -14.0}
        ]
        return {
            "final_duration_s": target,
            "timeline": timeline_ops,
            "narration": narration_lines,
            "subtitles": subtitles,
            "exports": exports,
        }

    if job_type == "dopamine_story":
        # Expect params: {"style": "reddit", "beats": 8, "platform": "tiktok"}
        beats = int(params.get("beats", 8))
        style = params.get("style", "reddit")
        lines = [
            "I never planned to post this...",
            "But last night everything changed.",
            "Short version? I messed up.",
            "And then it got worse.",
            "So I made a choice.",
            "It worked. Too well.",
            "Now I'm stuck with the truth.",
            "And you won't believe the ending."
        ][:beats]
        loops = [{"start": i*3.5, "end": i*3.5 + 3.0} for i in range(len(lines))]
        subtitles = [{"t_start": loops[i]["start"], "t_end": loops[i]["end"], "text": lines[i]} for i in range(len(lines))]
        return {
            "style": style,
            "script": lines,
            "visual_loops": loops,
            "voice": {"generated": True, "voice_id": "natural_v1"},
            "subtitles": subtitles,
            "export": {"platform": params.get("platform", "tiktok"), "url": "https://example.com/exports/dopamine_story/final.mp4"}
        }

    if job_type == "ai_script_writer":
        # Expect params: {"topic": str, "tone": str, "max_lines": int}
        topic = params.get("topic", "unexpected life lesson")
        tone = params.get("tone", "fast_paced")
        max_lines = int(params.get("max_lines", 10))
        lines = [
            f"Here's the setup: {topic}.",
            "Then a twist hits.",
            "We face the consequence.",
            "A bolder choice appears.",
            "Tension climbs fast.",
            "We take the leap.",
            "It pays off.",
            "One last surprise.",
            "Lesson lands clean.",
            "End with momentum."
        ][:max_lines]
        return {
            "topic": topic,
            "tone": tone,
            "pacing": "snappy",
            "beats": len(lines),
            "script": lines,
            "estimated_duration_s": len(lines) * 2.5
        }

    # default
    return {"message": "no-op"}


# -------------------- Video Upload Fix & Diagnostics --------------------
class UploadURLRequest(BaseModel):
    url: HttpUrl
    force: Optional[bool] = False
    # Optional raw Cookie header string to bypass YouTube bot/age checks
    cookie_header: Optional[str] = None

@app.post("/api/ingest/url")
async def ingest_by_url(payload: UploadURLRequest):
    # Validate first
    check = await validate_url(URLCheckRequest(url=payload.url))

    # If it's YouTube (text/html), run extraction layer to resolve real media stream
    extracted: Optional[ExtractedMedia] = None
    if (not check.ok and "text/html" in (check.content_type or "")) or _is_youtube(str(payload.url)):
        extracted = _extract_youtube_stream(str(payload.url), cookie_header=payload.cookie_header)
        if not extracted.ok:
            extra = " | Provide cookie_header from your logged-in browser if this persists." if extracted.needs_auth else ""
            raise HTTPException(status_code=400, detail=f"YouTube extraction failed: {extracted.reason}{extra}")
        # Force success with media details replacing HTML response
        check = URLCheckResponse(
            ok=True,
            reason=None,
            content_type=f"video/{extracted.container}" if extracted.container else "video/mp4",
            content_length=None,
        )

    if not check.ok and not payload.force:
        raise HTTPException(status_code=400, detail=f"Invalid video URL: {check.reason}")

    # Record video doc
    doc = {
        "source_type": "url",
        "source_url": str(payload.url),
        "status": "pending",
        "created_at": _now_ts(),
        "updated_at": _now_ts(),
        "error": None,
        "extracted": extracted.model_dump() if extracted else None,
    }
    vid = db["video"].insert_one(doc).inserted_id

    return {"video_id": str(vid), "status": "queued", "validation": check.model_dump(), "extracted": (extracted.model_dump() if extracted else None)}

@app.get("/api/diagnostics/upload")
async def upload_diagnostics(url: Optional[str] = None):
    """Quick diagnostics for URL parsing & reachability."""
    if not url:
        return {"ok": False, "reason": "No URL provided"}
    try:
        head = requests.head(url, allow_redirects=True, timeout=6)
        return {
            "ok": head.status_code < 400,
            "status": head.status_code,
            "headers": {k: v for k, v in head.headers.items() if k.lower() in ["content-type", "content-length"]},
        }
    except Exception as e:
        return {"ok": False, "reason": str(e)[:200]}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

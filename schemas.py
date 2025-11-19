"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
These schemas are used for data validation in your application.

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Product -> "product" collection
- BlogPost -> "blogs" collection
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Literal, Dict
from datetime import datetime

# -------------------- Core Domain Schemas --------------------

class Video(BaseModel):
    """
    Videos ingested into the system (by URL or file upload)
    Collection name: "video"
    """
    source_type: Literal["url", "upload"] = Field(..., description="How the video was provided")
    source_url: Optional[HttpUrl] = Field(None, description="URL if provided by link")
    file_name: Optional[str] = Field(None, description="Original filename if uploaded")
    duration_sec: Optional[float] = Field(None, ge=0)
    status: Literal["pending", "ready", "error"] = "pending"
    error: Optional[str] = None

class Job(BaseModel):
    """Generic processing job tracker. Collection name: "job"""
    video_id: Optional[str] = Field(None, description="Associated video id")
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
    ]
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    params: Dict = Field(default_factory=dict)
    result: Optional[Dict] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class Clip(BaseModel):
    """Extracted clip with quality score. Collection name: "clip"""
    video_id: str
    start_sec: float = Field(..., ge=0)
    end_sec: float = Field(..., ge=0)
    score: int = Field(..., ge=0, le=100)
    metrics: Dict = Field(default_factory=dict)
    highlights: List[str] = Field(default_factory=list)

class Thumbnail(BaseModel):
    """Thumbnail drafts. Collection name: "thumbnail"""
    video_id: str
    timecode_sec: float = Field(..., ge=0)
    text: str
    image_url: Optional[str] = None
    style: Literal["bold", "clean", "gaming_mono"] = "bold"

class SubtitleStyle(BaseModel):
    """Subtitle styling preferences per video/project. Collection name: "subtitlestyle"""
    video_id: str
    font_category: Literal["narration_clean", "reaction_bold", "gaming_mono"]
    font_family: str
    weight: Optional[int] = Field(None, ge=100, le=900)
    embed_in_video: bool = True

# Optional additional schemas for reference
class TrendProfile(BaseModel):
    platform: Literal["tiktok", "youtube_shorts"]
    pacing_ms_per_caption: int
    text_size_ratio: float
    voiceover_lead_ms: int

class MusicSuggestion(BaseModel):
    video_id: str
    track_id: str
    track_name: str
    bpm: Optional[int] = None
    gain_db: float = 0.0

# Example schemas (kept for reference)
class User(BaseModel):
    name: str
    email: str
    address: str
    age: Optional[int] = Field(None, ge=0, le=120)
    is_active: bool = True

class Product(BaseModel):
    title: str
    description: Optional[str] = None
    price: float = Field(..., ge=0)
    category: str
    in_stock: bool = True

# Note: The Flames database viewer will automatically:
# 1. Read these schemas from GET /schema endpoint
# 2. Use them for document validation when creating/editing
# 3. Handle all database operations (CRUD) directly
# 4. You don't need to create any database endpoints!

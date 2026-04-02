from typing import Dict, List, Optional

from pydantic import BaseModel


class MainFeature(BaseModel):
    label: str
    confidence_sum: float
    count: int


class StageLatency(BaseModel):
    caption_seconds: float
    detection_seconds: float
    depth_seconds: float


class FrameResult(BaseModel):
    frame: int
    narration: str
    action: str
    speak_now: bool
    main_feature: Optional[MainFeature] = None
    detected_objects: List[str]
    latency: StageLatency


class ProcessVideoResponse(BaseModel):
    filename: str
    narrations: List[str]
    frame_results: List[FrameResult]
    overall_main_feature: Optional[str] = None

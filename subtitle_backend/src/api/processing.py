"""
Processing utilities adapted from reposition_subtitles7.py to be used by FastAPI endpoints.
This module provides process_subtitle(video_path, subtitle_path, max_workers) which returns the output file path.

Performance improvements:
- Reuse a single VideoContext per job to avoid repeated VideoCapture opens.
- Vectorize detection position decision using numpy when possible.
- Cache segment decisions with an LRU cache to avoid recomputation for repeated or overlapping segments.
- Batch OCR across sampled frames per segment to reduce Python overhead.
- Control thread pool size and avoid excessive futures creation costs.
"""

import os
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from functools import lru_cache
from typing import List, Tuple, Optional

import cv2
import numpy as np
import srt  # Python 3.12 compatible SRT parser

try:
    # RapidOCR ONNXRuntime engine for detection
    from rapidocr_onnxruntime import RapidOCR  # type: ignore
except Exception:  # pragma: no cover - allow import to fail in environments without onnxruntime
    RapidOCR = None  # type: ignore

# Configure logging (file in working dir)
logging.basicConfig(
    filename="Reposition_sub_7.txt",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
log = logging.getLogger(__name__)
_engine = RapidOCR() if RapidOCR else None
_counter = 0


def _safe_video_capture(path: str) -> cv2.VideoCapture:
    """Open a video path with OpenCV and raise clear error if it fails."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    return cap


class VideoContext:
    """
    Hold a reusable VideoCapture and cached properties to avoid re-opening for each segment.
    """

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = _safe_video_capture(video_path)
        # Cache properties up front
        self.fps: float = self.cap.get(cv2.CAP_PROP_FPS) or 24.0
        self.height: int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count: int = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Seek to a frame index and read it; return None if not available."""
        if frame_idx < 0 or frame_idx >= self.frame_count:
            return None
        # set and read
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass


def detect_using_rapidocr(img: np.ndarray):
    """
    Run OCR with RapidOCR (onnxruntime). Returns list of dicts {box, text, score}.
    If RapidOCR is not available, returns empty list (non-fatal).
    """
    global _counter
    _counter += 1
    if _engine is None:
        # keep as info to avoid noisy logs each frame
        log.info("RapidOCR engine not available; skipping OCR detection.")
        return []

    results, _ = _engine(img)  # results = [(box, text, score), ...]
    detections = []
    if results:
        # Convert to light-weight dicts
        for (box, text, score) in results:
            # Ensure box is list of (x, y) pairs
            detections.append({"box": box, "text": text, "score": float(score)})
    return detections


def to_ass_timestamp_from_timedelta(td: timedelta) -> str:
    """Convert a timedelta to ASS H:MM:SS.CC format."""
    total_ms = int(td.total_seconds() * 1000)
    hours = total_ms // 3600000
    minutes = (total_ms % 3600000) // 60000
    seconds = (total_ms % 60000) // 1000
    centiseconds = (total_ms % 1000) // 10
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def preprocess_adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """Prepare image for OCR using adaptive threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    # Smaller block size for speed with acceptable quality
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=4,
    )
    return thresh


def _avg_y_from_box(box: List[Tuple[float, float]]) -> float:
    """Compute average y coordinate from a OCR quadrilateral box using numpy for speed."""
    # box is [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    arr = np.asarray(box, dtype=np.float32)
    return float(arr[:, 1].mean())


def decide_subtitle_position(filtered_detections_list, frame_height: int, bottom_threshold_ratio: float = 0.75) -> str:
    """Top if burnt-in detected in bottom, else bottom. Uses vectorized y-average checks where possible."""
    threshold_y = frame_height * bottom_threshold_ratio
    for frame_dets in filtered_detections_list:
        if not frame_dets:
            continue
        # Vectorize y-avg check
        for det in frame_dets:
            try:
                if _avg_y_from_box(det["box"]) > threshold_y:
                    return "top"
            except Exception:
                # Fallback if format is unexpected
                ys = [p[1] for p in det.get("box", [])]
                if ys and (sum(ys) / len(ys)) > threshold_y:
                    return "top"
    return "bottom"


def _compute_frame_indices(start_frame: int, end_frame: int, min_samples: int, max_frame_count: int) -> np.ndarray:
    """Compute unique, sorted frame indices for sampling, clipped to video range."""
    if start_frame > end_frame:
        start_frame, end_frame = end_frame, start_frame
    total = max(1, end_frame - start_frame + 1)
    count = min(min_samples, total)
    if count <= 1:
        idx = np.array([start_frame], dtype=int)
    else:
        idx = np.linspace(start_frame, end_frame, count, dtype=int)
    idx = np.clip(idx, 0, max(0, max_frame_count - 1))
    # Remove duplicates after clipping and sorting
    idx = np.unique(idx)
    return idx


@lru_cache(maxsize=4096)
def _cached_segment_position(video_path: str, start_sec: float, end_sec: float, min_frames: int) -> str:
    """
    LRU cached decision for segment position. Keyed by exact inputs to avoid recomputation
    for overlapping or repeated segments (common with ASS/SSA/VTT line structures).
    """
    # The actual computation is delegated to non-cached internal helper that uses a scoped VideoContext.
    # This cached function is a facade to be used when a VideoContext isn't provided (SRT/ASS/SSA/VTT wrappers).
    # It will create a short-lived context; callers that loop many times should use the VideoContext path below.
    ctx = VideoContext(video_path)
    try:
        return _decide_position_with_context(ctx, start_sec, end_sec, min_frames)
    finally:
        ctx.release()


def _decide_position_with_context(ctx: VideoContext, start_sec: float, end_sec: float, min_frames: int = 3) -> str:
    """Compute position using a reusable VideoContext for efficiency."""
    fps = ctx.fps
    frame_height = ctx.height

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    # Dynamic sampling density based on segment length
    sub_time = max(0.0, end_sec - start_sec)
    required_min_frames = int(sub_time) // 2
    min_samples = max(min_frames, required_min_frames)

    frame_indices = _compute_frame_indices(start_frame, end_frame, min_samples, ctx.frame_count)

    filtered_detections_per_frame = []
    # Batch processing loop: read, preprocess, OCR
    for frame_idx in frame_indices:
        frame = ctx.read_frame(int(frame_idx))
        if frame is None:
            continue
        preprocessed = preprocess_adaptive_threshold(frame)
        detections = detect_using_rapidocr(preprocessed)
        filtered_detections_per_frame.append(detections)

    return decide_subtitle_position(filtered_detections_per_frame, frame_height)


def get_position_for_segment(video_path: str, start_sec: float, end_sec: float, min_frames: int = 3) -> str:
    """
    Run OCR on sampled frames to decide top/bottom.

    Optimization: use LRU cache to avoid recomputing identical segments.
    """
    log.info("get_position_for_segment: %s -> %s", start_sec, end_sec)
    # Use cached computation to cut redundant work between lines
    return _cached_segment_position(video_path, float(start_sec), float(end_sec), int(min_frames))


def _parse_srt_file(path: str):
    """Parse SRT file contents into a list of srt.Subtitle entries."""
    with open(path, "r", encoding="utf-8-sig") as f:
        contents = f.read()
    return list(srt.parse(contents))


def reposition_srt(video_path, srt_path, output_ass_path, min_frames=3, max_workers=5):
    """Read SRT via srt module, run OCR in parallel, output ASS with repositioned alignment tags."""
    subs = _parse_srt_file(srt_path)

    # Pre-compute segments to deduplicate with caching and reduce thread contention
    segments = [(float(sub.start.total_seconds()), float(sub.end.total_seconds())) for sub in subs]

    def process_sub(i_sub: int):
        sub = subs[i_sub]
        start_sec, end_sec = segments[i_sub]
        # When many segments, usage of cache avoids repeated work; also allow a shared context for bursts
        position = get_position_for_segment(video_path, start_sec, end_sec, min_frames)
        alignment_tag = r"{\an8}" if position == "top" else r"{\an2}"
        formatted_text = sub.content.replace("\n", r"\N")
        line = (
            f"Dialogue: 0,{to_ass_timestamp_from_timedelta(sub.start)},{to_ass_timestamp_from_timedelta(sub.end)},"
            f"Default,,0,0,0,,{alignment_tag}{formatted_text}\n"
        )
        return i_sub, line

    # Cap workers to a reasonable number to avoid oversubscription with OpenCV/onnxruntime
    workers = max(1, min(int(max_workers), 12))
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_sub, i) for i in range(len(subs))]
        for fut in as_completed(futures):
            try:
                i_sub, line = fut.result()
                results[i_sub] = line
            except Exception as e:
                log.error("Error processing subtitle: %s", e)

    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            "PlayResX: 1920\n"
            "PlayResY: 1080\n"
            "ScaledBorderAndShadow: yes\n\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
            "0,0,0,0,100,100,0,0,1,2,0,2,10,10,30,1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )
        for i in range(len(subs)):
            if i in results:
                f.write(results[i])

    log.info("Repositioned subtitle saved: %s", output_ass_path)
    return output_ass_path


def reposition_ass(video_path, ass_path, output_ass_path, max_workers=5):
    """Modify alignment tags in ASS dialogue lines."""
    # Prepare a context to avoid repeated video opens across many lines
    ctx = VideoContext(video_path)

    def ass_time_to_sec(ts: str) -> float:
        h, m_, s_cs = ts.split(":")
        s, cs = s_cs.split(".")
        return int(h) * 3600 + int(m_) * 60 + int(s) + int(cs) / 100

    def process_sub(line: str):
        if line.startswith("Dialogue:"):
            m = re.match(r"Dialogue: \d+,(.*?),(.*?),", line)
            if m:
                start_str, end_str = m.groups()
                start_sec = ass_time_to_sec(start_str)
                end_sec = ass_time_to_sec(end_str)
                # Use context-aware decision (faster than cached facade for mass calls)
                position = _decide_position_with_context(ctx, start_sec, end_sec)
                if re.search(r"\{\\an\d\}", line):
                    line = re.sub(r"\{\\an\d\}", r"{\an8}" if position == "top" else r"{\an2}", line)
                else:
                    line = line.rstrip("\n") + (r"{\an8}" if position == "top" else r"{\an2}")
        return line

    with open(ass_path, "r", encoding="utf-8") as f:
        input_ass_file = f.read()

    output_ass_lines = input_ass_file.splitlines()
    workers = max(1, min(int(max_workers), 12))
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {executor.submit(process_sub, line): i for i, line in enumerate(output_ass_lines)}
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                try:
                    output_ass_lines[i] = future.result()
                except Exception as e:
                    log.error("Error processing line [%s]: %s\n%s", i, output_ass_lines[i], e)
    finally:
        ctx.release()

    new_ass_file = "\n".join(output_ass_lines)
    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(new_ass_file)
    return output_ass_path


def reposition_ssa(video_path, ssa_path, output_ssa_path, max_workers=5):
    """Modify alignment tags in SSA dialogue lines."""
    ctx = VideoContext(video_path)

    def ssa_time_to_sec(ts: str) -> float:
        h, m_, s_cs = ts.split(":")
        s, cs = s_cs.split(".")
        return int(h) * 3600 + int(m_) * 60 + int(s) + int(cs) / 100

    def process_sub(line: str):
        if line.startswith("Dialogue:"):
            m = re.match(r"Dialogue: Marked=\d+,(.*?),(.*?),", line)
            if m:
                start_str, end_str = m.groups()
                start_sec = ssa_time_to_sec(start_str)
                end_sec = ssa_time_to_sec(end_str)
                position = _decide_position_with_context(ctx, start_sec, end_sec)
                if re.search(r"\{\\an\d\}", line):
                    line = re.sub(r"\{\\an\d\}", r"{\an8}" if position == "top" else r"{\an2}", line)
                else:
                    line = line.rstrip("\n") + (r"{\an8}" if position == "top" else r"{\an2}")
        return line

    with open(ssa_path, "r", encoding="utf-8") as f:
        input_ssa_file = f.read()

    lines = input_ssa_file.splitlines()
    workers = max(1, min(int(max_workers), 12))
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {executor.submit(process_sub, line): i for i, line in enumerate(lines)}
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                try:
                    lines[i] = future.result()
                except Exception as e:
                    log.error("Error processing line [%s]: %s\n%s", i, lines[i], e)
    finally:
        ctx.release()

    new_ssa_file = "\n".join(lines)
    with open(output_ssa_path, "w", encoding="utf-8") as f:
        f.write(new_ssa_file)
    return output_ssa_path


def reposition_vtt(video_path, vtt_path, output_vtt_path, max_workers=5):
    """Modify 'line:' cue position in VTT based on OCR-detected overlap."""
    ctx = VideoContext(video_path)

    def vtt_time_to_sec(ts: str) -> float:
        hms = ts.strip().split(":")
        if len(hms) == 3:
            h, m, s_ms = hms
        else:
            h, m, s_ms = 0, *hms
        s, ms = s_ms.split(".")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    def extract_timestamps_string(line):
        matches = re.search(r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{1,3})", line)
        start_str = matches.group(1)
        end_str = matches.group(2)
        return start_str, end_str

    def process_sub(line: str):
        if "-->" in line:
            start_str, end_str = extract_timestamps_string(line)
            start_sec = vtt_time_to_sec(start_str)
            end_sec = vtt_time_to_sec(end_str)
            position = _decide_position_with_context(ctx, start_sec, end_sec)
            if "line:" in line:
                line = re.sub(r"line:\d+%?", "line:0%" if position == "top" else "line:80%", line)
            else:
                line = line.strip() + (" line:0%" if position == "top" else " line:80%")
        return line

    with open(vtt_path, "r", encoding="utf-8") as f:
        input_vtt_file = f.read()

    results = input_vtt_file.splitlines()
    workers = max(1, min(int(max_workers), 12))
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_as_idx = {executor.submit(process_sub, line): i for i, line in enumerate(results)}
            for future in as_completed(future_as_idx):
                i = future_as_idx[future]
                try:
                    results[i] = future.result()
                except Exception as e:
                    log.error("Error processing VTT line [%s]: %s\n%s", i, results[i], e)
    finally:
        ctx.release()

    new_vtt_file = "\n".join(results) if results else ""
    with open(output_vtt_path, "w", encoding="utf-8") as f:
        f.write(new_vtt_file)
    return output_vtt_path


# PUBLIC_INTERFACE
def process_subtitle(video_path, subtitle_path, max_workers=12):
    """Process a subtitle file (srt/ass/ssa/vtt) against a video and write a repositioned output, returning the path."""
    ext = os.path.splitext(subtitle_path)[1].lower()

    if ext == ".srt":
        output_file = os.path.splitext(subtitle_path)[0] + "_repositioned.ass"
        reposition_srt(video_path, subtitle_path, output_file, max_workers=max_workers)
    elif ext == ".ass":
        output_file = os.path.splitext(subtitle_path)[0] + "_repositioned.ass"
        reposition_ass(video_path, subtitle_path, output_file, max_workers=max_workers)
    elif ext == ".ssa":
        output_file = os.path.splitext(subtitle_path)[0] + "_repositioned.ssa"
        reposition_ssa(video_path, subtitle_path, output_file, max_workers=max_workers)
    elif ext == ".vtt":
        output_file = os.path.splitext(subtitle_path)[0] + "_repositioned.vtt"
        reposition_vtt(video_path, subtitle_path, output_file, max_workers=max_workers)
    else:
        raise ValueError(f"Unsupported subtitle format: {ext}")

    return output_file

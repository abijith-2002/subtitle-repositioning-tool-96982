import os
import cv2
import numpy as np
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Any

# Use pure-Python srt instead of pysrt (compat with Python 3.12, matches project deps)
import srt
from datetime import timedelta

# Try to import RapidOCR lazily and guard if unavailable
try:
    from rapidocr_onnxruntime import RapidOCR  # type: ignore
except Exception:  # pragma: no cover
    RapidOCR = None  # type: ignore

# Configure logging
logging.basicConfig(
    filename="Reposition_sub_7.txt",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
log = logging.getLogger(__name__)

# Limit OpenCV internal threading to avoid libavcodec/ffmpeg threading assert failures when combined with
# Python thread pools and onnxruntime threads. This mitigates "fctx->async_lock failed".
try:
    # Available on OpenCV 4.x
    cv2.setNumThreads(1)
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Lazily create OCR engine per process, not at import time, and avoid global reuse across threads.
# onnxruntime sessions can internally manage threads; creating per-call keeps lifecycle simple and avoids
# cross-thread access issues when used with ThreadPoolExecutor.
def _get_ocr_engine():
    return RapidOCR() if RapidOCR else None

_counter = 0


class VideoContext:
    """
    Reusable wrapper around cv2.VideoCapture to avoid re-opening video per segment.
    Provides cached properties and safe frame access.

    Note on crash root cause:
    The assertion "fctx->async_lock failed at libavcodec/pthread_frame.c:175" is typically triggered by
    unsafe interaction of FFmpeg's async frame/threaded decoding with multiple threading layers (OpenCV,
    Python ThreadPoolExecutor, and ONNXRuntime). This module mitigates it by:
      - Disabling OpenCV's internal threading (cv2.setNumThreads(1)).
      - Avoiding global shared ONNXRuntime session across threads (create per-call engine).
      - Capping pool sizes.
      - Removing pysrt (C-extension) in favor of pure-Python 'srt' to reduce ABI/runtime conflicts.
    """

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        self.fps: float = float(self.cap.get(cv2.CAP_PROP_FPS) or 24.0)
        self.height: int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count: int = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Seek to a frame index and return the frame, or None if not available."""
        if frame_idx < 0 or frame_idx >= self.frame_count:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass


def detect_using_rapidocr(img: np.ndarray) -> List[Dict[str, Any]]:
    """Run OCR with ONNXRuntime over a preprocessed image and normalize results. Returns [] if OCR unavailable."""
    global _counter
    _counter += 1
    engine = _get_ocr_engine()
    if engine is None:
        if _counter <= 3:
            log.info("RapidOCR engine not available; skipping OCR detection.")
        return []
    try:
        results, _ = engine(img)  # results = [(box, text, score), ...]
    except Exception as e:
        log.error("RapidOCR inference failed: %s", e)
        return []
    detections: List[Dict[str, Any]] = []
    if results:
        for (box, text, score) in results:
            detections.append({"box": box, "text": text, "score": float(score)})
    return detections


def _ass_ts_from_timedelta(td: timedelta) -> str:
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
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=5,
    )
    return thresh


def decide_subtitle_position(filtered_detections_list, frame_height, bottom_threshold_ratio=0.75):
    """Top if burnt-in detected in bottom, else bottom."""
    for frame_detections in filtered_detections_list:
        if frame_detections:
            for det in frame_detections:
                y_coords = [p[1] for p in det.get("box", [])]
                if not y_coords:
                    continue
                avg_y = sum(y_coords) / len(y_coords)
                if avg_y > frame_height * bottom_threshold_ratio:
                    return "top"
    return "bottom"


def _compute_frame_indices(start_frame: int, end_frame: int, min_frames: int, max_frame_count: int) -> np.ndarray:
    """Compute sorted unique frame indices within bounds."""
    if start_frame > end_frame:
        start_frame, end_frame = end_frame, start_frame
    total = max(1, end_frame - start_frame + 1)
    count = min(max(1, int(min_frames)), total)
    idx = np.linspace(start_frame, end_frame, count, dtype=int)
    idx = np.clip(idx, 0, max(0, max_frame_count - 1))
    return np.unique(idx)


def get_position_for_segment_ctx(ctx: VideoContext, start_sec: float, end_sec: float, min_frames: int = 3) -> str:
    """Compute position using a reusable VideoContext."""
    fps = ctx.fps
    frame_height = ctx.height
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    # Dynamically increase sampling for longer segments
    sub_time = max(0.0, end_sec - start_sec)
    required_min_frames = int(sub_time) // 2
    min_frames = max(int(min_frames), int(required_min_frames))

    frame_indices = _compute_frame_indices(start_frame, end_frame, min_frames, ctx.frame_count)
    filtered_detections_per_frame = []
    for frame_idx in frame_indices:
        frame = ctx.read_frame(int(frame_idx))
        if frame is None:
            log.info("could not obtain frame at idx %s", frame_idx)
            continue
        preprocessed = preprocess_adaptive_threshold(frame)
        detections = detect_using_rapidocr(preprocessed)
        filtered_detections_per_frame.append(detections)

    log.info("filtered detections per frame count: %s", len(filtered_detections_per_frame))
    return decide_subtitle_position(filtered_detections_per_frame, frame_height)


def get_position_for_segment(video_path, start_sec, end_sec, min_frames=3):
    """Run OCR on sampled frames to decide top/bottom using a short-lived context (compat shim)."""
    ctx = VideoContext(video_path)
    try:
        return get_position_for_segment_ctx(ctx, start_sec, end_sec, min_frames)
    finally:
        ctx.release()


def _parse_srt_file(path: str) -> List[srt.Subtitle]:
    """Parse SRT file contents into a list of srt.Subtitle entries."""
    with open(path, "r", encoding="utf-8-sig") as f:
        contents = f.read()
    return list(srt.parse(contents))

def reposition_srt(video_path, srt_path, output_ass_path, results, min_frames=3, max_workers=5):
    """Read SRT and write ASS using provided results, no video access here."""
    subs = _parse_srt_file(srt_path)

    # Map of index to constructed line ensures stable ordering
    def process_sub(i_sub: int, position: str):
        sub = subs[i_sub]
        alignment_tag = r"{\an8}" if position == "top" else r"{\an2}"
        formatted_text = sub.content.replace("\n", r"\N")
        line = (
            f"Dialogue: 0,{_ass_ts_from_timedelta(sub.start)},{_ass_ts_from_timedelta(sub.end)},"
            f"Default,,0,0,0,,{alignment_tag}{formatted_text}\n"
        )
        return i_sub, line

    # Build lookup: results is a dict keyed by caller conventions (index->info) here
    idx_to_pos: Dict[int, str] = {}
    for key, val in results.items():
        if isinstance(val, dict) and "subtitle_index" in val and "recommended_position" in val:
            idx_to_pos[int(val["subtitle_index"])] = str(val["recommended_position"])

    workers = max(1, min(int(max_workers), 8))
    lines_out: Dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for i in range(len(subs)):
            pos = idx_to_pos.get(i, "bottom")
            futures.append(executor.submit(process_sub, i, pos))
        for fut in as_completed(futures):
            i_sub, line = fut.result()
            lines_out[i_sub] = line

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
            if i in lines_out:
                f.write(lines_out[i])

    log.info("Repositioned subtitle saved: %s", output_ass_path)
    return output_ass_path


import traceback
def reposition_ass(ass_path, results, output_ass_path, max_workers=5):
    """Modify only alignment tags in ASS dialogue lines using provided results; no video access here."""
    with open(ass_path, "r") as f:
        input_ass_lines = f.read().splitlines()
        log.info("Loaded ASS lines: %d", len(input_ass_lines))
    output_ass_lines = input_ass_lines[:]

    def process_sub(sub_index, position):
        line = input_ass_lines[sub_index]
        if line.startswith("Dialogue:"):
            m = re.match(r"Dialogue: \d+,(.*?),(.*?),", line)
            if m:
                # Replace or insert alignment tag using provided position
                if re.search(r"\{\\an\d\}", line):
                    line = re.sub(r"\{\\an\d\}", r"{\\an8}" if position == "top" else r"{\\an2}", line)
                else:
                    line = line.rstrip("\n") + (r"{\an8}" if position == "top" else r"{\an2}")
                output_ass_lines[sub_index] = line
        return line

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                process_sub,
                results[result].get("subtitle_index"),
                results[result].get("recommended_position"),
            ): results[result].get("subtitle_index")
            for result in results
        }
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            try:
                _ = future.result()
                log.info("Processed ASS line [%s]", i)
            except Exception as e:
                log.error(f"Error processing line '{input_ass_lines[i]}', Error: {e}")
                traceback.print_exc()

    new_ass_file = "\n".join(output_ass_lines) if output_ass_lines else ""
    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(new_ass_file)
    return output_ass_path


def reposition_ssa(ssa_path, results, output_ssa_path, max_workers=5):
    """Modify only alignment tags in SSA dialogue lines using provided results; no video access here."""
    with open(ssa_path, "r", encoding="utf-8") as f:
        input_ssa_lines = f.read().splitlines()
        log.info("Loaded SSA lines: %d", len(input_ssa_lines))
    output_ssa_lines = input_ssa_lines[:]

    def process_sub(sub_index, position):
        line = input_ssa_lines[sub_index]
        if line.startswith("Dialogue:"):
            m = re.match(r"Dialogue: Marked=\d+,(.*?),(.*?),", line)
            if m:
                if re.search(r"\{\\an\d\}", line):
                    line = re.sub(r"\{\\an\d\}", r"{\\an8}" if position == "top" else r"{\\an2}", line)
                else:
                    line = line.rstrip("\n") + (r"{\an8}" if position == "top" else r"{\an2}")
                output_ssa_lines[sub_index] = line
        return line

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_as_idx = {
            executor.submit(
                process_sub,
                results[result].get("subtitle_index"),
                results[result].get("recommended_position"),
            ): results[result].get("subtitle_index")
            for result in results
        }
        for future in as_completed(future_as_idx):
            i = future_as_idx[future]
            try:
                _ = future.result()
                log.info("Processed SSA line [%s]", i)
            except Exception as e:
                log.error(f"Error processing line: {input_ssa_lines[i]}: {e}")

    new_ssa_file = "\n".join(output_ssa_lines) if output_ssa_lines else ""
    with open(output_ssa_path, "w", encoding="utf-8") as f:
        f.write(new_ssa_file)
    return output_ssa_path


def reposition_vtt(vtt_path, results, output_vtt_path, max_workers=5):
    """Modify 'line:' cue position in VTT using provided results; no video access here."""
    log.info("repositioning vtt file")
    with open(vtt_path, "r", encoding="utf-8") as f:
        input_vtt_lines = f.read().splitlines()
        log.info("Loaded VTT lines: %d", len(input_vtt_lines))
    output_vtt_lines = input_vtt_lines[:]

    def vtt_time_to_sec(ts: str) -> float:
        hms = ts.strip().split(":")
        if len(hms) == 3:
            h, m, s_ms = hms
        else:
            h, m, s_ms = 0, *hms
        s, ms = s_ms.split(".")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    def extract_timestamps_string(line: str):
        matches = re.search(r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{1,3})", line)
        start_str = matches.group(1)
        end_str = matches.group(2)
        return start_str, end_str

    def process_sub(sub_index, position):
        line = input_vtt_lines[sub_index]
        if "-->" in line:
            start_str, end_str = extract_timestamps_string(line)
            _ = vtt_time_to_sec(start_str)  # parsed but unused in this stage
            _ = vtt_time_to_sec(end_str)
            if "line:" in line:
                line = re.sub(r"line:\d+%?", "line:0%" if position == "top" else "line:80%", line)
            else:
                line = line.strip() + (" line:0%" if position == "top" else " line:80%")
            output_vtt_lines[sub_index] = line
        return line

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_as_idx = {
            executor.submit(
                process_sub,
                results[result].get("subtitle_index"),
                results[result].get("recommended_position"),
            ): results[result].get("subtitle_index")
            for result in results
        }
        for future in as_completed(future_as_idx):
            i = future_as_idx[future]
            try:
                _ = future.result()
                log.info("Processed VTT line [%s]", i)
            except Exception as e:
                log.error(f"Error processing VTT line [{i}]: {e}", exc_info=True)

    new_vtt_file = "\n".join(output_vtt_lines) if output_vtt_lines else ""
    with open(output_vtt_path, "w", encoding="utf-8") as f:
        f.write(new_vtt_file)
    return output_vtt_path


def get_detections_ctx(ctx: VideoContext, start_sec: float, end_sec: float, min_frames: int = 3) -> Dict[str, Any]:
    """Collect detection analysis using a reusable VideoContext."""
    fps = ctx.fps
    frame_height = ctx.height
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    sub_time = max(0.0, end_sec - start_sec)
    required_min_frames = int(sub_time) // 2
    min_frames = max(int(min_frames), int(required_min_frames))

    frame_indices = _compute_frame_indices(start_frame, end_frame, min_frames, ctx.frame_count)
    analysis = []
    filtered_detections_per_frame = []
    for frame_idx in frame_indices:
        rec: Dict[str, Any] = {}
        frame = ctx.read_frame(int(frame_idx))
        if frame is None:
            log.info("could not obtain frame at idx %s", frame_idx)
            continue
        preprocessed = preprocess_adaptive_threshold(frame)
        detections = detect_using_rapidocr(preprocessed)
        filtered_detections_per_frame.append(detections)
        rec["frame_index"] = int(frame_idx)
        rec["timestamp"] = float(frame_idx / fps)
        rec["detections"] = detections
        analysis.append(rec)

    recommended_position = decide_subtitle_position(filtered_detections_per_frame, frame_height)
    return {"analysis": analysis, "recommended_position": recommended_position}


def get_detections(video_path, start_sec, end_sec, min_frames=3):
    """Compatibility shim: open a short-lived context and delegate to get_detections_ctx."""
    ctx = VideoContext(video_path)
    try:
        return get_detections_ctx(ctx, start_sec, end_sec, min_frames)
    finally:
        ctx.release()


def detect_text_srt(video_path, srt_path, min_frames=3, max_workers=5):
    """Read SRT and collect detections using a single VideoContext for all segments."""
    subs = _parse_srt_file(srt_path)
    ctx = VideoContext(video_path)

    def process_one(i_sub: int):
        sub = subs[i_sub]
        start_sec = float(sub.start.total_seconds())
        end_sec = float(sub.end.total_seconds())
        log.info(f"sub text:{sub.content}")
        detections = get_detections_ctx(ctx, start_sec, end_sec, min_frames)
        detections["subtitle_index"] = i_sub
        return i_sub, detections

    try:
        workers = max(1, min(int(max_workers), 4))
        results: Dict[int, Any] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_one, i) for i in range(len(subs))]
            for fut in as_completed(futures):
                i_sub, det = fut.result()
                results[i_sub] = det
        return results
    finally:
        ctx.release()


def detect_text_ass(video_path, ass_path, max_workers=5):
    """Parse ASS and collect detections using a single VideoContext for all segments."""
    ctx = VideoContext(video_path)

    def process_sub(line: str, sub_index: int):
        if line.startswith("Dialogue:"):
            m = re.match(r"Dialogue: \d+,(.*?),(.*?),", line)
            if m:
                start_str, end_str = m.groups()

                def ass_time_to_sec(ts: str) -> float:
                    h, m_, s_cs = ts.split(":")
                    s, cs = s_cs.split(".")
                    return int(h) * 3600 + int(m_) * 60 + int(s) + int(cs) / 100

                start_sec = ass_time_to_sec(start_str)
                end_sec = ass_time_to_sec(end_str)
                matches = re.search(
                    r"Dialogue: \d+,[0-9]:[0-9]{2}:[0-9]{2}\.\d+,[0-9]{1,2}:[0-9]{2}:[0-9]{2}\.\d+,(?:Default)?,(?:.*)?,\d+,\d+,\d+,(?:.*)?,(?:\{\\an\d\})?(.*)",
                    line,
                )
                sub_text = matches.groups()[0] if matches else ""
                log.info(f"sub text:{sub_text}")
                detections = get_detections_ctx(ctx, start_sec, end_sec)
                detections["subtitle_index"] = sub_index
                return detections
        return None

    with open(ass_path, "r") as f:
        input_ass_file = f.read()
        log.info("input_ass_file loaded")
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(process_sub, line, i): i for i, line in enumerate(input_ass_file.splitlines())}
            results: Dict[int, Any] = {}
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                try:
                    result = future.result()
                    log.info(f"original [{i}] : {input_ass_file.splitlines()[i]}")
                    log.info(f"result [{i}] : {result}")
                    if result:
                        results[i] = result
                except Exception as e:
                    log.error(f"Error processing line '{input_ass_file.splitlines()[i]}',Error:{e}")
                    traceback.print_exc()
        return results
    finally:
        ctx.release()


def detect_text_ssa(video_path, ssa_path, max_workers=5):
    """Parse SSA and collect detections using a single VideoContext for all segments."""
    ctx = VideoContext(video_path)

    def process_sub(line: str, sub_index: int):
        if line.startswith("Dialogue:"):
            m = re.match(r"Dialogue: Marked=\d+,(.*?),(.*?),", line)
            if m:
                start_str, end_str = m.groups()

                def ssa_time_to_sec(ts: str) -> float:
                    h, m_, s_cs = ts.split(":")
                    s, cs = s_cs.split(".")
                    return int(h) * 3600 + int(m_) * 60 + int(s) + int(cs) / 100

                start_sec = ssa_time_to_sec(start_str)
                end_sec = ssa_time_to_sec(end_str)
                matches = re.search(
                    r"Dialogue: Marked=\d+,[0-9]:[0-9]{2}:[0-9]{2}\.\d+,[0-9]{1,2}:[0-9]{2}:[0-9]{2}\.\d+,(?:Default)?,(?:.*)?,\d+,\d+,\d+,,(?:\{\\an\d\})?(.*)",
                    line,
                )
                sub_text = matches.groups()[0] if matches else ""
                log.info(f"sub text: {sub_text}")
                detections = get_detections_ctx(ctx, start_sec, end_sec)
                detections["subtitle_index"] = sub_index
                return detections
        return None

    with open(ssa_path, "r", encoding="utf-8") as f:
        input_ssa_file = f.read()
        log.info("input_ssa_file loaded")
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_as_idx = {executor.submit(process_sub, line, i): i for i, line in enumerate(input_ssa_file.splitlines())}
            results: Dict[int, Any] = {}
            k = 0
            for future in as_completed(future_as_idx):
                i = future_as_idx[future]
                try:
                    result = future.result()
                    log.info(f"original[{i}] : {input_ssa_file.splitlines()[i]}")
                    log.info(f"result [{i}] : {result}")
                    if result:
                        results[k] = result
                        k += 1
                except Exception:
                    log.error(f"Error processing line: {input_ssa_file.splitlines()[i]}")
        return results
    finally:
        ctx.release()


def detect_text_vtt(video_path, vtt_path, max_workers=5):
    """Parse VTT and collect detections using a single VideoContext for all segments."""
    log.info("processing vtt file for detections")
    ctx = VideoContext(video_path)
    min_frames = 3

    def vtt_time_to_sec(ts: str) -> float:
        hms = ts.strip().split(":")
        if len(hms) == 3:
            h, m, s_ms = hms
        else:
            h, m, s_ms = 0, *hms
        s, ms = s_ms.split(".")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    def extract_timestamps_string(line: str):
        matches = re.search(r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{1,3})", line)
        start_str = matches.group(1)
        end_str = matches.group(2)
        return start_str, end_str

    def process_sub(line: str, sub_index: int):
        if "-->" in line:
            start_str, end_str = extract_timestamps_string(line)
            start_sec = vtt_time_to_sec(start_str)
            end_sec = vtt_time_to_sec(end_str)
            detections = get_detections_ctx(ctx, start_sec, end_sec, min_frames)
            detections["subtitle_index"] = sub_index
            return detections
        return None

    with open(vtt_path, "r", encoding="utf-8") as f:
        input_vtt_file = f.read()
        log.info("input_vtt_file loaded")
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_as_idx = {executor.submit(process_sub, line, i): i for i, line in enumerate(input_vtt_file.splitlines())}
            results: Dict[int, Any] = {}
            for future in as_completed(future_as_idx):
                i = future_as_idx[future]
                try:
                    result = future.result()
                    log.info(f"original [{i}] : {input_vtt_file.splitlines()[i]}")
                    log.info(f"result  [{i}] : {result}")
                    if result:
                        results[i] = result
                except Exception as e:
                    log.error(f"Error processing line: {input_vtt_file.splitlines()[i]} ({e})", exc_info=True)
        return results
    finally:
        ctx.release()


def display_results(results):
    results = dict(sorted(results.items()))
    for index, result in results.items():
        log.info(f"index:{index}")
        log.info(f"result:{result}")


import time
# PUBLIC_INTERFACE
def process_subtitle(video_path: str, subtitle_path: str, max_workers: int = 8) -> str:
    """Process a subtitle file against a video and write a repositioned output.

    Mitigations for ffmpeg/libavcodec assertion:
    - OpenCV threads limited to 1 (cv2.setNumThreads(1)).
    - Avoid global shared RapidOCR engine across threads; create per-call engines.
    - Cap worker threads to a reasonable number (<=8, internally often <=4) to reduce contention.

    Returns path to the output file.
    """
    start = time.time()
    ext = os.path.splitext(subtitle_path)[1].lower()
    print("in process subtitle")
    if ext == ".srt":
        results = detect_text_srt(video_path, subtitle_path, max_workers=max_workers)
        log.info(f"results:{results}")
        if results:
            display_results(results)
        output_file = os.path.splitext(subtitle_path)[0] + "_repositioned.ass"
        reposition_srt(video_path, subtitle_path, output_file, results=results, max_workers=max_workers)
    elif ext == ".ass":
        results = detect_text_ass(video_path, subtitle_path, max_workers=max_workers)
        if results:
            display_results(results)
        output_file = os.path.splitext(subtitle_path)[0] + "_repositioned.ass"
        reposition_ass(ass_path=subtitle_path, output_ass_path=output_file, results=results, max_workers=max_workers)
    elif ext == ".ssa":
        results = detect_text_ssa(video_path, subtitle_path, max_workers=max_workers)
        if results:
            display_results(results)
        output_file = os.path.splitext(subtitle_path)[0] + "_repositioned.ssa"
        reposition_ssa(ssa_path=subtitle_path, output_ssa_path=output_file, results=results, max_workers=max_workers)
    elif ext == ".vtt":
        results = detect_text_vtt(video_path, subtitle_path, max_workers=max_workers)
        if results:
            display_results(results)
        output_file = os.path.splitext(subtitle_path)[0] + "_repositioned.vtt"
        reposition_vtt(vtt_path=subtitle_path, output_vtt_path=output_file, results=results, max_workers=max_workers)
    else:
        raise ValueError(f"Unsupported subtitle format: {ext}")

    log.info(f"Repositioned subtitle saved: {output_file}")
    print(f"Repositioned subtitle saved: {output_file}")
    end = time.time()
    print("total time taken", end - start)
    return output_file

if __name__ == "__main__":
    # Example paths; adjust as needed for local testing (use provided sample assets)
    video_path = "subtitle_backend/Key_and_Peele_sample1.mp4"
    sub_path = "subtitle_backend/Key_and_Peele_sample1.ass"
    max_workers = 6
    process_subtitle(video_path, sub_path, max_workers)

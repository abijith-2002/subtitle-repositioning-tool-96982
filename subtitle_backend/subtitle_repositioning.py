import os
import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

# Use pure-Python SRT parser (compatible with Python 3.12) instead of pysrt
import srt

try:
    from rapidocr_onnxruntime import RapidOCR  # type: ignore
except Exception:
    RapidOCR = None  # type: ignore

# Configure logging with moderate verbosity
logging.basicConfig(
    filename="Reposition_sub_7.txt",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
log = logging.getLogger(__name__)

# Initialize OCR engine once (if available)
_engine = RapidOCR() if RapidOCR else None
_frame_counter = 0


def _safe_video_capture(path: str) -> cv2.VideoCapture:
    """Open video with OpenCV and verify it opened successfully."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    return cap


class _VideoContext:
    """Reusable video context to avoid reopening for each segment."""

    def __init__(self, video_path: str):
        self.path = video_path
        self.cap = _safe_video_capture(video_path)
        self.fps: float = self.cap.get(cv2.CAP_PROP_FPS) or 24.0
        self.height: int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count: int = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frame(self, idx: int) -> Optional[np.ndarray]:
        if idx < 0 or idx >= self.frame_count:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
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
    """Run OCR with RapidOCR (onnxruntime). Returns list of dicts. Fallback to empty if engine missing."""
    global _frame_counter
    _frame_counter += 1
    if _engine is None:
        # Keep log low to avoid per-frame spam
        return []
    results, _ = _engine(img)
    detections = []
    if results:
        for (box, text, score) in results:
            detections.append({"box": box, "text": text, "score": float(score)})
    return detections


def _to_ass_timestamp_from_seconds(seconds: float) -> str:
    """Convert seconds to ASS H:MM:SS.CC format."""
    total_ms = int(seconds * 1000)
    hours = total_ms // 3600000
    minutes = (total_ms % 3600000) // 60000
    secs = (total_ms % 60000) // 1000
    centiseconds = (total_ms % 1000) // 10
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def _to_ass_timestamp_from_timedelta(start, end) -> Tuple[str, str]:
    """Helper for srt.Subtitle start/end timedelta to ASS strings."""
    return _to_ass_timestamp_from_seconds(start.total_seconds()), _to_ass_timestamp_from_seconds(end.total_seconds())


def preprocess_adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """Prepare image for OCR using adaptive threshold (fast and robust enough)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=4,
    )
    return thresh


def _avg_y_from_box(box) -> float:
    arr = np.asarray(box, dtype=np.float32)
    return float(arr[:, 1].mean()) if arr.size else 0.0


def decide_subtitle_position(filtered_detections_list: List[List[dict]], frame_height: int, bottom_threshold_ratio: float = 0.75) -> str:
    """Top if burnt-in detected in bottom region; else bottom."""
    threshold_y = frame_height * bottom_threshold_ratio
    for frame_dets in filtered_detections_list:
        if not frame_dets:
            continue
        for det in frame_dets:
            try:
                if _avg_y_from_box(det["box"]) > threshold_y:
                    return "top"
            except Exception:
                ys = [p[1] for p in det.get("box", [])]
                if ys and (sum(ys) / len(ys)) > threshold_y:
                    return "top"
    return "bottom"


def _compute_frame_indices(start_frame: int, end_frame: int, min_samples: int, max_count: int) -> np.ndarray:
    if start_frame > end_frame:
        start_frame, end_frame = end_frame, start_frame
    total = max(1, end_frame - start_frame + 1)
    count = min(min_samples, total)
    if count <= 1:
        idx = np.array([start_frame], dtype=int)
    else:
        idx = np.linspace(start_frame, end_frame, count, dtype=int)
    idx = np.clip(idx, 0, max(0, max_count - 1))
    return np.unique(idx)


def _decide_position_with_context(ctx: _VideoContext, start_sec: float, end_sec: float, min_frames: int = 3) -> str:
    fps = ctx.fps
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    # Dynamic sampling: proportional to segment length
    sub_time = max(0.0, end_sec - start_sec)
    required_min = max(min_frames, int(sub_time) // 2)

    indices = _compute_frame_indices(start_frame, end_frame, required_min, ctx.frame_count)
    filtered = []
    for fi in indices:
        frame = ctx.read_frame(int(fi))
        if frame is None:
            continue
        pre = preprocess_adaptive_threshold(frame)
        dets = detect_using_rapidocr(pre)
        filtered.append(dets)
    return decide_subtitle_position(filtered, ctx.height)


@lru_cache(maxsize=4096)
def _cached_position(video_path: str, start_sec: float, end_sec: float, min_frames: int) -> str:
    """LRU cache to avoid recomputation for identical segments."""
    ctx = _VideoContext(video_path)
    try:
        return _decide_position_with_context(ctx, start_sec, end_sec, min_frames)
    finally:
        ctx.release()


def get_position_for_segment(video_path, start_sec, end_sec, min_frames=3):
    """Run OCR on sampled frames to decide top/bottom (cached)."""
    log.info("get_position_for_segment: %s -> %s", start_sec, end_sec)
    return _cached_position(video_path, float(start_sec), float(end_sec), int(min_frames))


def _parse_srt_file(path: str) -> List[srt.Subtitle]:
    with open(path, "r", encoding="utf-8-sig") as f:
        contents = f.read()
    return list(srt.parse(contents))


def reposition_srt(video_path, srt_path, output_ass_path, results, min_frames=3, max_workers=5):
    """Read SRT, apply provided positions, output ASS with repositioned alignment tags (no OCR here)."""
    subs = _parse_srt_file(srt_path)

    # Build quick index by subtitle index to avoid searching the list each time
    index_to_sub = {i + 1: sub for i, sub in enumerate(subs)}

    def process_sub(sub_index: int, position: str) -> Tuple[int, str]:
        sub = index_to_sub.get(sub_index)
        if sub is None:
            return sub_index, ""
        start_str, end_str = _to_ass_timestamp_from_timedelta(sub.start, sub.end)
        alignment_tag = r"{\an8}" if position == "top" else r"{\an2}"
        formatted_text = sub.content.replace("\n", r"\N")
        line = f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{alignment_tag}{formatted_text}\n"
        return sub_index, line

    # Submit only once per provided result entry
    lines_out: Dict[int, str] = {}
    workers = max(1, min(int(max_workers), 12))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        fut_to_key = {}
        for key in results:
            entry = results[key]
            sub_idx = entry.get("subtitle_index")
            pos = entry.get("recommended_position")
            if sub_idx is None or pos is None:
                continue
            fut = executor.submit(process_sub, int(sub_idx), str(pos))
            fut_to_key[fut] = key

        for fut in as_completed(fut_to_key):
            idx, line = fut.result()
            if line:
                lines_out[idx] = line

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
        # preserve original order by iterating subs
        for i, sub in enumerate(subs, start=1):
            line = lines_out.get(i)
            if line:
                f.write(line)
    log.info("Repositioned subtitle saved: %s", output_ass_path)
    return output_ass_path


def reposition_ass(ass_path, results, output_ass_path, max_workers=5):
    """Modify only alignment tags in ASS dialogue lines using precomputed detection results."""
    with open(ass_path, "r", encoding="utf-8") as f:
        input_lines = f.read().splitlines()
    output_lines = list(input_lines)

    def process_by_index(sub_index: int, position: str) -> Tuple[int, str]:
        if sub_index < 0 or sub_index >= len(input_lines):
            return sub_index, ""
        line = input_lines[sub_index]
        if line.startswith("Dialogue:"):
            if re.search(r"\{\\an\d\}", line):
                line = re.sub(r"\{\\an\d\}", r"{\\an8}" if position == "top" else r"{\\an2}", line)
            else:
                line = line.rstrip("\n") + (r"{\an8}" if position == "top" else r"{\an2}")
        return sub_index, line

    workers = max(1, min(int(max_workers), 12))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        fut_to_key = {}
        for key, entry in results.items():
            sub_idx = entry.get("subtitle_index")
            pos = entry.get("recommended_position")
            if sub_idx is None or pos is None:
                continue
            fut = executor.submit(process_by_index, int(sub_idx), str(pos))
            fut_to_key[fut] = key

        for fut in as_completed(fut_to_key):
            try:
                idx, new_line = fut.result()
                if new_line:
                    output_lines[idx] = new_line
            except Exception as e:
                log.error("Error updating ASS line for key %s: %s", fut_to_key[fut], e, exc_info=True)

    new_text = "\n".join(output_lines)
    with open(output_ass_path, "w", encoding="utf-8") as fo:
        fo.write(new_text)


def reposition_ssa(ssa_path, results, output_ssa_path, max_workers=5):
    """Modify only alignment tags in SSA dialogue lines using precomputed detection results."""
    with open(ssa_path, "r", encoding="utf-8") as f:
        input_lines = f.read().splitlines()
    output_lines = list(input_lines)

    def process_by_index(sub_index: int, position: str) -> Tuple[int, str]:
        if sub_index < 0 or sub_index >= len(input_lines):
            return sub_index, ""
        line = input_lines[sub_index]
        if line.startswith("Dialogue:"):
            if re.search(r"\{\\an\d\}", line):
                line = re.sub(r"\{\\an\d\}", r"{\\an8}" if position == "top" else r"{\\an2}", line)
            else:
                line = line.rstrip("\n") + (r"{\an8}" if position == "top" else r"{\an2}")
        return sub_index, line

    workers = max(1, min(int(max_workers), 12))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        fut_to_key = {}
        for key, entry in results.items():
            sub_idx = entry.get("subtitle_index")
            pos = entry.get("recommended_position")
            if sub_idx is None or pos is None:
                continue
            fut = executor.submit(process_by_index, int(sub_idx), str(pos))
            fut_to_key[fut] = key

        for fut in as_completed(fut_to_key):
            try:
                idx, updated_line = fut.result()
                if updated_line:
                    output_lines[idx] = updated_line
            except Exception as e:
                log.error("Error updating SSA line for key %s: %s", fut_to_key[fut], e, exc_info=True)

    new_text = "\n".join(output_lines)
    with open(output_ssa_path, "w", encoding="utf-8") as f:
        f.write(new_text)


def reposition_vtt(vtt_path, results, output_vtt_path, max_workers=5):
    """For VTT: modify 'line:' cue position using precomputed detection results."""
    with open(vtt_path, "r", encoding="utf-8") as f:
        input_lines = f.read().splitlines()
    output_lines = list(input_lines)

    def process_by_index(sub_index: int, position: str) -> Tuple[int, str]:
        if sub_index < 0 or sub_index >= len(input_lines):
            return sub_index, ""
        line = input_lines[sub_index]
        if "-->" in line:
            if "line:" in line:
                line = re.sub(r"line:\d+%?", "line:0%" if position == "top" else "line:80%", line)
            else:
                line = line.strip() + (" line:0%" if position == "top" else " line:80%")
        return sub_index, line

    workers = max(1, min(int(max_workers), 12))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        fut_to_key = {}
        for key, entry in results.items():
            sub_idx = entry.get("subtitle_index")
            pos = entry.get("recommended_position")
            if sub_idx is None or pos is None:
                continue
            fut = executor.submit(process_by_index, int(sub_idx), str(pos))
            fut_to_key[fut] = key

        for fut in as_completed(fut_to_key):
            try:
                idx, new_line = fut.result()
                if new_line:
                    output_lines[idx] = new_line
            except Exception as e:
                log.error("Error updating VTT line for key %s: %s", fut_to_key[fut], e, exc_info=True)

    new_text = "\n".join(output_lines)
    with open(output_vtt_path, "w", encoding="utf-8") as f:
        f.write(new_text)


def _srt_time_to_seconds(sub: srt.Subtitle) -> Tuple[float, float]:
    return float(sub.start.total_seconds()), float(sub.end.total_seconds())


def get_detections(video_path, start_sec, end_sec, min_frames=3):
    """Run OCR on sampled frames, return per-frame analysis and recommended position."""
    ctx = _VideoContext(video_path)
    try:
        fps = ctx.fps
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        sub_time = max(0.0, end_sec - start_sec)
        required_min = max(int(min_frames), int(sub_time) // 2)
        indices = _compute_frame_indices(start_frame, end_frame, required_min, ctx.frame_count)
        analysis = []
        filtered = []
        for fi in indices:
            entry = {"frame_index": int(fi), "timestamp": float(fi / fps), "detections": []}
            frame = ctx.read_frame(int(fi))
            if frame is None:
                analysis.append(entry)
                continue
            pre = preprocess_adaptive_threshold(frame)
            dets = detect_using_rapidocr(pre)
            filtered.append(dets)
            entry["detections"] = dets
            analysis.append(entry)
        recommended = decide_subtitle_position(filtered, ctx.height)
        return {"analysis": analysis, "recommended_position": recommended}
    finally:
        ctx.release()


def detect_text_srt(video_path, srt_path, min_frames=3, max_workers=5):
    """Analyze SRT cues and compute recommended positions using OCR (parallelized)."""
    subs = _parse_srt_file(srt_path)

    def process_sub(sub: srt.Subtitle, sub_index: int):
        start_sec, end_sec = _srt_time_to_seconds(sub)
        log.info("sub text: %s", sub.content)
        det = get_detections(video_path, start_sec, end_sec, min_frames)
        det["subtitle_index"] = sub_index
        log.info("detections%s", det)
        return det

    workers = max(1, min(int(max_workers), 12))
    results: Dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        fut_to_idx = {executor.submit(process_sub, sub, i + 1): i for i, sub in enumerate(subs)}
        for fut in as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                log.error("Error processing subtitle %s: %s", idx, e)
                results[idx] = None  # keep structure consistent
    return results


def _ass_time_to_sec(ts: str) -> float:
    h, m_, s_cs = ts.split(":")
    s, cs = s_cs.split(".")
    return int(h) * 3600 + int(m_) * 60 + int(s) + int(cs) / 100


def detect_text_ass(video_path, ass_path, max_workers=5):
    """Compute detections for ASS Dialogue lines."""
    def process_line(line: str):
        if not line.startswith("Dialogue:"):
            return None
        m = re.match(r"Dialogue: \d+,(.*?),(.*?),", line)
        if not m:
            return None
        start_str, end_str = m.groups()
        start_sec = _ass_time_to_sec(start_str)
        end_sec = _ass_time_to_sec(end_str)
        det = get_detections(video_path, start_sec, end_sec)
        return det

    with open(ass_path, "r", encoding="utf-8") as f:
        content = f.read()
    lines = content.splitlines()

    workers = max(1, min(int(max_workers), 12))
    results: Dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        fut_to_idx = {executor.submit(process_line, line): i for i, line in enumerate(lines)}
        for fut in as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            try:
                det = fut.result()
                if det:
                    results[i] = det
            except Exception as e:
                log.error("Error processing line [%s]: %s", i, e, exc_info=True)
    return results


def _ssa_time_to_sec(ts: str) -> float:
    h, m_, s_cs = ts.split(":")
    s, cs = s_cs.split(".")
    return int(h) * 3600 + int(m_) * 60 + int(s) + int(cs) / 100


def detect_text_ssa(video_path, ssa_path, max_workers=5):
    """Compute detections for SSA Dialogue lines."""
    def process_line(line: str):
        if not line.startswith("Dialogue:"):
            return None
        m = re.match(r"Dialogue: Marked=\d+,(.*?),(.*?),", line)
        if not m:
            return None
        start_str, end_str = m.groups()
        start_sec = _ssa_time_to_sec(start_str)
        end_sec = _ssa_time_to_sec(end_str)
        det = get_detections(video_path, start_sec, end_sec)
        return det

    with open(ssa_path, "r", encoding="utf-8") as f:
        content = f.read()
    lines = content.splitlines()

    workers = max(1, min(int(max_workers), 12))
    results: Dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        fut_to_idx = {executor.submit(process_line, line): i for i, line in enumerate(lines)}
        for fut in as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            try:
                det = fut.result()
                if det:
                    results[i] = det
            except Exception:
                log.error("Error processing line: %s", lines[i])
    return results


def _vtt_time_to_sec(ts: str) -> float:
    parts = ts.strip().split(":")
    if len(parts) == 3:
        h, m, s_ms = parts
    else:
        h, m, s_ms = 0, *parts
    s, ms = s_ms.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def detect_text_vtt(video_path, vtt_path, max_workers=5):
    """Compute detections for VTT cue lines."""
    def extract_ts(line: str):
        m = re.search(r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{1,3})", line)
        if not m:
            return None
        return m.group(1), m.group(2)

    def process_line(line: str):
        if "-->" not in line:
            return None
        ts = extract_ts(line)
        if not ts:
            return None
        start_str, end_str = ts
        start_sec = _vtt_time_to_sec(start_str)
        end_sec = _vtt_time_to_sec(end_str)
        det = get_detections(video_path, start_sec, end_sec)
        return det

    with open(vtt_path, "r", encoding="utf-8") as f:
        content = f.read()
    lines = content.splitlines()

    workers = max(1, min(int(max_workers), 12))
    results: Dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        fut_to_idx = {executor.submit(process_line, line): i for i, line in enumerate(lines)}
        for fut in as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            try:
                det = fut.result()
                if det:
                    results[i] = det
            except Exception as e:
                log.error("Error processing line [%s]: %s", i, e, exc_info=True)
    return results


def display_results(results: Dict[int, dict]):
    """Sort and log results (reduced verbosity)."""
    for index in sorted(results.keys()):
        log.info("index:%s", index)
        log.info("result:%s", results[index])


def process_subtitle(video_path, subtitle_path, max_workers=12):
    """Process subtitle file and write repositioned output. Preserves original behavior."""
    start = time.time()
    ext = os.path.splitext(subtitle_path)[1].lower()
    print("in process subtitle")

    if ext == ".srt":
        results = detect_text_srt(video_path, subtitle_path, max_workers=max_workers)
        if results:
            display_results(results)
        else:
            print("No results")
        output_file = os.path.splitext(subtitle_path)[0] + "_repositioned.ass"
        reposition_srt(video_path, subtitle_path, output_file, results=results, max_workers=max_workers)

    elif ext == ".ass":
        results = detect_text_ass(video_path, subtitle_path, max_workers=max_workers)
        if results:
            display_results(results)
        else:
            print("No results")
        output_file = os.path.splitext(subtitle_path)[0] + "_repositioned.ass"
        reposition_ass(ass_path=subtitle_path, results=results, output_ass_path=output_file, max_workers=max_workers)

    elif ext == ".ssa":
        results = detect_text_ssa(video_path, subtitle_path, max_workers=max_workers)
        if results:
            display_results(results)
        else:
            print("No results")
        output_file = os.path.splitext(subtitle_path)[0] + "_repositioned.ssa"
        reposition_ssa(ssa_path=subtitle_path, results=results, output_ssa_path=output_file, max_workers=max_workers)

    elif ext == ".vtt":
        results = detect_text_vtt(video_path, subtitle_path, max_workers=max_workers)
        if results:
            display_results(results)
        else:
            print("No results")
        output_file = os.path.splitext(subtitle_path)[0] + "_repositioned.vtt"
        reposition_vtt(vtt_path=subtitle_path, results=results, output_vtt_path=output_file, max_workers=max_workers)

    else:
        raise ValueError(f"Unsupported subtitle format: {ext}")

    log.info("Repositioned subtitle saved: %s", output_file)
    print(f"Repositioned subtitle saved: {output_file}")
    end = time.time()
    print("total time taken", end - start)
    return output_file


if __name__ == "__main__":
    # Example direct run (paths are placeholders for local testing)
    video_path = r"uploads\Key_and_Peele_sample1.mp4"
    sub_path = r"outputs\Key_and_Peele_sample1.ass"
    max_workers = 10
    process_subtitle(video_path, sub_path, max_workers)

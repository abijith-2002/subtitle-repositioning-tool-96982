import os
import cv2
import numpy as np
import pysrt
import re
from rapidocr_onnxruntime import RapidOCR  # OCR engine
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Any

# Configure logging
logging.basicConfig(
    filename="Reposition_sub_7.txt",
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
log = logging.getLogger(__name__)

# Initialize OCR engine once
_engine = RapidOCR()
_counter = 0


class VideoContext:
    """
    Reusable wrapper around cv2.VideoCapture to avoid re-opening video per segment.
    Provides cached properties and safe frame access.
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
    """Run OCR with ONNXRuntime over a preprocessed image and normalize results."""
    global _counter
    _counter += 1
    log.info("OCR frame counter: %s", _counter)
    results, _ = _engine(img)
    detections: List[Dict[str, Any]] = []
    if results:
        log.info("detections found for frame")
        for (box, text, score) in results:
            detections.append({"box": box, "text": text, "score": float(score)})
    return detections


def to_ass_timestamp(srt_time):
    """Convert pysrt.SubRipTime to ASS H:MM:SS.CC format."""
    total_ms = (
        srt_time.hours * 3600 * 1000
        + srt_time.minutes * 60 * 1000
        + srt_time.seconds * 1000
        + srt_time.milliseconds
    )
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


def reposition_srt(video_path, srt_path, output_ass_path, results, min_frames=3, max_workers=5):
    """Read SRT and write ASS using provided results, no video access here."""
    subs = pysrt.open(srt_path)

    def process_sub(sub_index, position):
        sub = next((s for s in subs if s.index == sub_index), None)
        if not sub:
            return ""
        log.info("sub obtained")
        log.info(f"sub text:{sub.text}")
        alignment_tag = r"{\an8}" if position == "top" else r"{\an2}"
        formatted_text = sub.text.replace("\n", r"\N")
        line = (
            f"Dialogue: 0,{to_ass_timestamp(sub.start)},{to_ass_timestamp(sub.end)},"
            f"Default,,0,0,0,,{alignment_tag}{formatted_text}\n"
        )
        return line

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                process_sub,
                results[result].get("subtitle_index"),
                results[result].get("recommended_position"),
            ): i
            for i, result in enumerate(results)
        }
        results_map: Dict[int, Optional[str]] = {}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results_map[idx] = future.result()
            except Exception as e:
                log.error(f"Error processing subtitle {idx}: {e}")
                results_map[idx] = None

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
            if i in results_map and results_map[i]:
                f.write(results_map[i])

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
    subs = pysrt.open(srt_path)
    ctx = VideoContext(video_path)

    def process_sub(sub, sub_index):
        start_sec = (
            sub.start.hours * 3600
            + sub.start.minutes * 60
            + sub.start.seconds
            + sub.start.milliseconds / 1000
        )
        end_sec = (
            sub.end.hours * 3600
            + sub.end.minutes * 60
            + sub.end.seconds
            + sub.end.milliseconds / 1000
        )
        log.info(f"sub text:{sub.text}")
        detections = get_detections_ctx(ctx, start_sec, end_sec, min_frames)
        detections["subtitle_index"] = sub_index
        return detections

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(process_sub, sub, sub.index): i for i, sub in enumerate(subs)}
            results: Dict[int, Any] = {}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                    log.info("retrieved result")
                except Exception as e:
                    log.error(f"Error processing subtitle {idx}: {e}")
                    results[idx] = None
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
def process_subtitle(video_path, subtitle_path, max_workers=12):
    start = time.time()
    ext = os.path.splitext(subtitle_path)[1].lower()
    print("in process subtitle")
    if ext == ".srt":
        results = detect_text_srt(video_path, subtitle_path, max_workers=max_workers)
        log.info(f"results:{results}")
        if results:
            display_results(results)
        else:
            print("No results")
        output_file = os.path.splitext(subtitle_path)[0] + "_repositioned.ass"
        reposition_srt(video_path, subtitle_path, output_file, results=results, max_workers=max_workers)
    elif ext == ".ass":
        results = detect_text_ass(video_path, subtitle_path, max_workers=max_workers)
        print("displaying results")
        if results:
            display_results(results)
        else:
            print("No results")
        output_file = os.path.splitext(subtitle_path)[0] + "_repositioned.ass"
        reposition_ass(ass_path=subtitle_path, output_ass_path=output_file, results=results, max_workers=max_workers)
    elif ext == ".ssa":
        results = detect_text_ssa(video_path, subtitle_path, max_workers=max_workers)
        log.info(f"results :\n{results}\n type(results):{type(results)}")
        if results:
            display_results(results)
        else:
            print("No results")
        output_file = os.path.splitext(subtitle_path)[0] + "_repositioned.ssa"
        reposition_ssa(ssa_path=subtitle_path, output_ssa_path=output_file, results=results, max_workers=max_workers)
    elif ext == ".vtt":
        results = detect_text_vtt(video_path, subtitle_path, max_workers=max_workers)
        log.info(results)
        if results:
            display_results(results)
        else:
            print("No results")
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
    # Example paths; adjust as needed for local testing
    video_path = r"uploads\Key_and_Peele_sample1.mp4"
    sub_path = r"outputs\Key_and_Peele_sample1.ass"
    max_workers = 10
    process_subtitle(video_path, sub_path, max_workers)

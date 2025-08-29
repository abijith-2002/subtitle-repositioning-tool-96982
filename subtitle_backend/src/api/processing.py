"""
Processing utilities adapted from reposition_subtitles7.py to be used by FastAPI endpoints.
This module provides process_subtitle(video_path, subtitle_path, max_workers) which returns the output file path.
"""

import os
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pysrt

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


def detect_using_rapidocr(img):
    """
    Run OCR with RapidOCR (onnxruntime). Returns list of dicts {box, text, score}.
    If RapidOCR is not available, returns empty list (non-fatal).
    """
    global _counter
    _counter += 1
    if _engine is None:
        log.warning("RapidOCR engine not available; skipping OCR detection.")
        return []

    results, _ = _engine(img)  # results = [(box, text, score), ...]
    detections = []
    if results:
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


def preprocess_adaptive_threshold(image):
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
                y_coords = [p[1] for p in det["box"]]
                avg_y = sum(y_coords) / len(y_coords)
                if avg_y > frame_height * bottom_threshold_ratio:
                    return "top"
    return "bottom"


def get_position_for_segment(video_path, start_sec, end_sec, min_frames=3):
    """Run OCR on sampled frames to decide top/bottom."""
    log.info("get_position_for_segment: %s -> %s", start_sec, end_sec)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    sub_time = max(0.0, end_sec - start_sec)
    required_min_frames = int(sub_time) // 2
    min_frames = max(min_frames, required_min_frames)

    total = max(1, abs(end_frame - start_frame + 1))
    frame_indices = np.linspace(start_frame, end_frame, min(min_frames, total), dtype=int)

    filtered_detections_per_frame = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            log.info("could not obtain frame at index %s", frame_idx)
            continue
        preprocessed = preprocess_adaptive_threshold(frame)
        detections = detect_using_rapidocr(preprocessed)
        filtered_detections_per_frame.append(detections)
    cap.release()

    return decide_subtitle_position(filtered_detections_per_frame, frame_height)


def reposition_srt(video_path, srt_path, output_ass_path, min_frames=3, max_workers=5):
    """Read SRT, run OCR in parallel, output ASS with repositioned alignment tags."""
    subs = pysrt.open(srt_path)

    def process_sub(sub):
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
        position = get_position_for_segment(video_path, start_sec, end_sec, min_frames)
        alignment_tag = r"{\an8}" if position == "top" else r"{\an2}"
        formatted_text = sub.text.replace("\n", r"\N")
        line = (
            f"Dialogue: 0,{to_ass_timestamp(sub.start)},{to_ass_timestamp(sub.end)},"
            f"Default,,0,0,0,,{alignment_tag}{formatted_text}\n"
        )
        return line

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(process_sub, sub): i for i, sub in enumerate(subs)}
        results = {}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                log.error("Error processing subtitle %s: %s", idx, e)
                results[idx] = None

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
            if results.get(i):
                f.write(results[i])

    log.info("Repositioned subtitle saved: %s", output_ass_path)
    return output_ass_path


def reposition_ass(video_path, ass_path, output_ass_path, max_workers=5):
    """Modify alignment tags in ASS dialogue lines."""

    def process_sub(line: str):
        if line.startswith("Dialogue:"):
            m = re.match(r"Dialogue: \d+,(.*?),(.*?),", line)
            if m:
                start_str, end_str = m.groups()

                def ass_time_to_sec(ts):
                    h, m_, s_cs = ts.split(":")
                    s, cs = s_cs.split(".")
                    return int(h) * 3600 + int(m_) * 60 + int(s) + int(cs) / 100

                start_sec = ass_time_to_sec(start_str)
                end_sec = ass_time_to_sec(end_str)
                position = get_position_for_segment(video_path, start_sec, end_sec)

                if re.search(r"\{\\an\d\}", line):
                    line = re.sub(r"\{\\an\d\}", r"{\an8}" if position == "top" else r"{\an2}", line)
                else:
                    line = line.rstrip("\n") + (r"{\an8}" if position == "top" else r"{\an2}")
        return line

    with open(ass_path, "r", encoding="utf-8") as f:
        input_ass_file = f.read()

    output_ass_lines = input_ass_file.splitlines()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(process_sub, line): i for i, line in enumerate(output_ass_lines)}
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            try:
                output_ass_lines[i] = future.result()
            except Exception as e:
                log.error("Error processing line [%s]: %s\n%s", i, output_ass_lines[i], e)

    new_ass_file = "\n".join(output_ass_lines)
    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(new_ass_file)
    return output_ass_path


def reposition_ssa(video_path, ssa_path, output_ssa_path, max_workers=5):
    """Modify alignment tags in SSA dialogue lines."""

    def process_sub(line: str):
        if line.startswith("Dialogue:"):
            m = re.match(r"Dialogue: Marked=\d+,(.*?),(.*?),", line)
            if m:
                start_str, end_str = m.groups()

                def ssa_time_to_sec(ts):
                    h, m_, s_cs = ts.split(":")
                    s, cs = s_cs.split(".")
                    return int(h) * 3600 + int(m_) * 60 + int(s) + int(cs) / 100

                start_sec = ssa_time_to_sec(start_str)
                end_sec = ssa_time_to_sec(end_str)
                position = get_position_for_segment(video_path, start_sec, end_sec)

                if re.search(r"\{\\an\d\}", line):
                    line = re.sub(r"\{\\an\d\}", r"{\an8}" if position == "top" else r"{\an2}", line)
                else:
                    line = line.rstrip("\n") + (r"{\an8}" if position == "top" else r"{\an2}")
        return line

    with open(ssa_path, "r", encoding="utf-8") as f:
        input_ssa_file = f.read()

    lines = input_ssa_file.splitlines()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(process_sub, line): i for i, line in enumerate(lines)}
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            try:
                lines[i] = future.result()
            except Exception as e:
                log.error("Error processing line [%s]: %s\n%s", i, lines[i], e)

    new_ssa_file = "\n".join(lines)
    with open(output_ssa_path, "w", encoding="utf-8") as f:
        f.write(new_ssa_file)
    return output_ssa_path


def reposition_vtt(video_path, vtt_path, output_vtt_path, max_workers=5):
    """Modify 'line:' cue position in VTT based on OCR-detected overlap."""

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
            position = get_position_for_segment(video_path, start_sec, end_sec)
            if "line:" in line:
                line = re.sub(r"line:\d+%?", "line:0%" if position == "top" else "line:80%", line)
            else:
                line = line.strip() + (" line:0%" if position == "top" else " line:80%")
        return line

    with open(vtt_path, "r", encoding="utf-8") as f:
        input_vtt_file = f.read()

    results = input_vtt_file.splitlines()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_as_idx = {executor.submit(process_sub, line): i for i, line in enumerate(results)}
        for future in as_completed(future_as_idx):
            i = future_as_idx[future]
            try:
                results[i] = future.result()
            except Exception as e:
                log.error("Error processing VTT line [%s]: %s\n%s", i, results[i], e)

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

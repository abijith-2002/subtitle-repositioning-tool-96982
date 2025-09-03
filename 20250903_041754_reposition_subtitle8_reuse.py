import os
import cv2
import numpy as np
import pysrt
import re
from typing import Dict, Tuple, Any, List

# from rapidocr_test import detect_using_rapidocr
from rapidocr_onnxruntime import RapidOCR  # for detection using rapidocr_onnxruntime
# from rapidocr import RapidOCR # for detection using rapidocr
import logging

logging.basicConfig(
    filename="Reposition_sub_7.txt",
    filemode='w',
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding='utf-8'
)
log = logging.getLogger()
engine = RapidOCR()
a = 0

# Global cache to reuse initial get_text_* results.
# Keys are (round(start_sec, 3), round(end_sec, 3)) to normalize floating jitter.
# Values are dicts returned by get_detections(...) including "recommended position".
DETECTION_CACHE: Dict[Tuple[float, float], Dict[str, Any]] = {}

def _norm_key(start_sec: float, end_sec: float) -> Tuple[float, float]:
    """Normalize time window to a stable cache key."""
    return (round(float(start_sec), 3), round(float(end_sec), 3))

def detect_using_rapidocr(img):
    global a
    print(f"a:{a}")
    log.info(f"a:{a}")

    # Run OCR with ONNXRuntime
    results, _ = engine(img)  # results = [(box, text, score), ...]

    detections = []
    if results:
        log.info("detections found for frame")
        for i, (box, text, score) in enumerate(results):
            temp_result = {
                "box": box,
                "text": text,
                "score": float(score)
            }
            detections.append(temp_result)
    a += 1
    return detections

def to_ass_timestamp(srt_time):
    """Convert pysrt.SubRipTime to ASS H:MM:SS.CC format."""
    total_ms = (
        srt_time.hours * 3600 * 1000 +
        srt_time.minutes * 60 * 1000 +
        srt_time.seconds * 1000 +
        srt_time.milliseconds
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
        C=5
    )
    return thresh

def decide_subtitle_position(filtered_detections_list, frame_height, bottom_threshold_ratio=0.75):
    """Top if burnt-in detected in bottom, else bottom."""
    for frame_detections in filtered_detections_list:
        if frame_detections:
            for det in frame_detections:
                y_coords = [p[1] for p in det["box"]]
                avg_y = sum(y_coords) / len(y_coords) if y_coords else 0
                if avg_y > frame_height * bottom_threshold_ratio:
                    return "top"
    return "bottom"

def get_position_for_segment(video_path, start_sec, end_sec, min_frames=3):
    """Run OCR on sampled frames to decide top/bottom.
    Modified to first check the DETECTION_CACHE for a precomputed decision.
    """
    key = _norm_key(start_sec, end_sec)
    cached = DETECTION_CACHE.get(key)
    if cached and isinstance(cached, dict) and cached.get("recommended position"):
        # Reuse cached decision from initial get_text_* run
        log.info(f"Using cached position for segment {key}: {cached.get('recommended position')}")
        return cached.get("recommended position")

    # Fallback: run detection (for standalone usage if detect_text_* was not called)
    log.info("in get position for segment (no cache) ")
    log.info(f"start sec:{start_sec},end_sec:{end_sec}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    sub_time = end_sec - start_sec
    required_min_frames = int(sub_time) // 2
    min_frames = max(min_frames, required_min_frames)
    frame_indices = np.linspace(start_frame, end_frame, min(min_frames, abs(end_frame - start_frame + 1)), dtype=int)

    filtered_detections_per_frame = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            log.info("could not obtain frame")
            continue
        preprocessed = preprocess_adaptive_threshold(frame)
        detections = detect_using_rapidocr(preprocessed)
        filtered_detections_per_frame.append(detections)
    cap.release()
    return decide_subtitle_position(filtered_detections_per_frame, frame_height)

from concurrent.futures import ThreadPoolExecutor, as_completed

def reposition_srt(video_path, srt_path, output_ass_path, min_frames=3, max_workers=5):
    """Read SRT, reuse cached decisions from detect_text_srt, and output ASS with repositioned alignment tags."""
    subs = pysrt.open(srt_path)

    def process_sub(sub):
        start_sec = sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds + sub.start.milliseconds / 1000
        end_sec = sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds + sub.end.milliseconds / 1000
        key = _norm_key(start_sec, end_sec)
        # Prefer cached result
        cached = DETECTION_CACHE.get(key)
        if cached and cached.get("recommended position"):
            position = cached["recommended position"]
            log.info(f"SRT reuse cached decision for {key}: {position}")
        else:
            # Fallback (should be rare if detect_text_srt was called first)
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
                log.error(f"Error processing subtitle {idx}: {e}")
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

    log.info(f"Repositioned subtitle saved: {output_ass_path}")
    return output_ass_path

import traceback
def reposition_ass(video_path, ass_path, output_ass_path , max_workers = 5):
    """Modify only alignment tags in ASS/SSA dialogue lines using cached results from detect_text_ass."""
    def process_sub(line:str):
        if line.startswith("Dialogue:"):
            m = re.match(r"Dialogue: \d+,(.*?),(.*?),", line)
            if m:
                start_str, end_str = m.groups()
                def ass_time_to_sec(ts):
                    h, m_, s_cs = ts.split(":")
                    s, cs = s_cs.split(".")
                    return int(h)*3600 + int(m_)*60 + int(s) + int(cs)/100
                start_sec = ass_time_to_sec(start_str)
                end_sec = ass_time_to_sec(end_str)
                key = _norm_key(start_sec, end_sec)

                # Prefer cached decision if available
                cached = DETECTION_CACHE.get(key)
                if cached and cached.get("recommended position"):
                    position = cached["recommended position"]
                    log.info(f"ASS reuse cached decision for {key}: {position}")
                else:
                    position = get_position_for_segment(video_path, start_sec, end_sec)

                if re.search(r"\{\\an\d\}", line):
                    line = re.sub(r"\{\\an\d\}", r"{\an8}" if position == "top" else r"{\an2}", line)
                else:
                    line = line.rstrip("\n") + (r"{\an8}" if position == "top" else r"{\an2}")
        return line

    with open(ass_path,'r',encoding="utf-8") as f:
        input_ass_file = f.read()
        log.info(f"input_ass_file:\n{input_ass_file}")

    output_ass_file = input_ass_file.splitlines()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(process_sub,line):i
            for i,line in enumerate(input_ass_file.splitlines())
            }
        for future in as_completed(future_to_idx):
            i= future_to_idx[future]
            try:
                result = future.result()
                log.info(f"original [{i}] : {input_ass_file.splitlines()[i]}")
                log.info(f"result [{i}] : {result}")
                output_ass_file[int(i)] = result
                log.info(f"completed[{i}]")
            except Exception as e:
                log.error(f"Error processing line '{input_ass_file.splitlines()[i]}',Error:{e}")
                traceback.print_exc()
        log.info(f"output_ass_file:{output_ass_file}")
    new_ass_file = "\n".join(output_ass_file) if output_ass_file else ""
    log.info(new_ass_file)
    open(output_ass_path,"w",encoding="utf-8").write(new_ass_file)

def reposition_ssa(video_path, ssa_path, output_ssa_path , max_workers =5 ):
    """Modify only alignment tags in SSA dialogue lines using cached results from detect_text_ssa."""
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
                key = _norm_key(start_sec, end_sec)

                cached = DETECTION_CACHE.get(key)
                if cached and cached.get("recommended position"):
                    position = cached["recommended position"]
                    log.info(f"SSA reuse cached decision for {key}: {position}")
                else:
                    position = get_position_for_segment(video_path, start_sec, end_sec)

                if re.search(r"\{\\an\d\}", line):
                    line = re.sub(r"\{\\an\d\}", r"{\an8}" if position == "top" else r"{\an2}", line)
                else:
                    line = line.rstrip("\n") + (r"{\an8}" if position == "top" else r"{\an2}")
        return line

    with open(ssa_path, 'r', encoding="utf-8") as f:
        input_ssa_file = f.read()
        log.info(f"input_ssa_file:\n{input_ssa_file}")
    results = input_ssa_file.splitlines()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_as_idx =  { executor.submit(process_sub,line):i for i,line in enumerate(input_ssa_file.splitlines())}
        for future in as_completed(future_as_idx):
            i = future_as_idx[future]
            try:
                result = future.result()
                log.info(f"original[{i}] : {input_ssa_file.splitlines()[i]}")
                log.info(f"result [{i}] : {result}")
                results[i] = result
            except:
                log.error(f"Error processing line: {input_ssa_file.splitlines()[i]}")
    new_ssa_file = "\n".join(results) if results else ""
    log.info(new_ssa_file)
    with open(output_ssa_path, "w", encoding="utf-8") as f:
        f.write(new_ssa_file)

def reposition_vtt(video_path, vtt_path, output_vtt_path , max_workers = 5):
    """For VTT: modify 'line:' cue position like ASS/SSA alignment using cached results from detect_text_vtt."""
    log.info("repositioning vtt file")

    def vtt_time_to_sec(ts: str) -> float:
        hms = ts.strip().split(":")
        if len(hms) == 3:  # hh:mm:ss.mmm
            h, m, s_ms = hms
        else:  # mm:ss.mmm
            h, m, s_ms = 0, *hms
        s, ms = s_ms.split(".")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    def extract_timestamps_string(line):
        matches = re.search(r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{1,3})",line)
        start_str = matches.group(1)
        end_str = matches.group(2)
        return start_str , end_str

    def process_sub(line: str):
        if "-->" in line:  # This is a cue timing line
            start_str , end_str = extract_timestamps_string(line)
            start_sec = vtt_time_to_sec(start_str)
            end_sec = vtt_time_to_sec(end_str)
            key = _norm_key(start_sec, end_sec)

            cached = DETECTION_CACHE.get(key)
            if cached and cached.get("recommended position"):
                position = cached["recommended position"]
                log.info(f"VTT reuse cached decision for {key}: {position}")
            else:
                position = get_position_for_segment(video_path, start_sec, end_sec)

            if "line:" in line:
                line = re.sub(r"line:\d+%?",
                              "line:0%" if position == "top" else "line:80%",
                              line)
            else:
                line = line.strip() + (r" line:0%" if position == "top" else r" line:80%")
        return line

    with open(vtt_path, "r", encoding="utf-8") as f:
        input_vtt_file = f.read()
        log.info(f"input_vtt_file:\n{input_vtt_file}")
    results = input_vtt_file.splitlines()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_as_idx = {executor.submit(process_sub,line):i for i,line in enumerate(input_vtt_file.splitlines())}
        for future in as_completed(future_as_idx):
            i = future_as_idx[future]
            try:
                result = future.result()
                log.info(f"original [{i}] : {input_vtt_file.splitlines()[i]}")
                log.info(f"result  [{i}] : {result}")
                results[i] = result
            except Exception as e:
                log.error(f"Error processing line: {input_vtt_file.splitlines()[i]} ({e})")
                log.error("error occured:",exc_info=True)

    new_vtt_file = "\n".join(results) if results else ""
    log.info(new_vtt_file)
    with open(output_vtt_path, "w", encoding="utf-8") as f:
        f.write(new_vtt_file)

#----------------------------------------------------printing out detections (initial run)---------------------------------------------------

def get_detections(video_path, start_sec, end_sec, min_frames=3):
    """Run OCR on sampled frames and produce analysis + recommended position.
    Results are stored into DETECTION_CACHE for reuse.
    """
    key = _norm_key(start_sec, end_sec)
    # If already computed, return cached
    if key in DETECTION_CACHE:
        log.info(f"get_detections: using cache for {key}")
        return DETECTION_CACHE[key]

    log.info("in get position for segment: generating new detections")
    log.info(f"start sec:{start_sec},end_sec:{end_sec}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    sub_time = end_sec-start_sec
    required_min_frames = int(sub_time)//2
    min_frames = max(min_frames,required_min_frames)
    frame_indices = np.linspace(start_frame, end_frame, min(min_frames, abs(end_frame - start_frame + 1)), dtype=int)

    analysis = []
    filtered_detections_per_frame = []
    for frame_idx in frame_indices:
        result = {}
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            log.info("could not obtain frame")
            continue
        preprocessed = preprocess_adaptive_threshold(frame)
        detections = detect_using_rapidocr(preprocessed)
        filtered_detections_per_frame.append(detections)
        result["frame_index"] = int(frame_idx)
        result["timestamp"] = float(frame_idx/fps)
        result["detections"] = detections
        analysis.append(result)
    log.info(f"filtered detections per frame{filtered_detections_per_frame}")
    cap.release()
    recommended_position= decide_subtitle_position(filtered_detections_per_frame, frame_height)
    d = {"analysis":analysis,"recommended position":recommended_position}
    # Store into global cache for reuse by reposition_* functions
    DETECTION_CACHE[key] = d
    return d

def detect_text_srt(video_path, srt_path, min_frames=3, max_workers=5):
    """Read SRT and compute detections for each cue. Stores results in DETECTION_CACHE for reuse."""
    subs = pysrt.open(srt_path)

    def process_sub(sub):
        start_sec = sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds + sub.start.milliseconds / 1000
        end_sec = sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds + sub.end.milliseconds / 1000
        log.info(f"sub text:{sub.text}")
        detections = get_detections(video_path,start_sec,end_sec,min_frames)
        log.info(f"detections{detections}")
        return ( _norm_key(start_sec,end_sec), detections )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(process_sub, sub): i for i, sub in enumerate(subs)}
        results = {}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                key, det = future.result()
                # ensure cache populated
                DETECTION_CACHE[key] = det
                results[idx] = det
                log.info("retrieved result")
            except Exception as e:
                log.error(f"Error processing subtitle {idx}: {e}")
                results[idx] = None
    return results

def detect_text_ass(video_path, ass_path, max_workers = 5):
    """Parse ASS dialogue, compute detections and populate DETECTION_CACHE."""
    def process_sub(line:str):
        if line.startswith("Dialogue:"):
            m = re.match(r"Dialogue: \d+,(.*?),(.*?),", line)
            if m:
                start_str, end_str = m.groups()
                def ass_time_to_sec(ts):
                    h, m_, s_cs = ts.split(":")
                    s, cs = s_cs.split(".")
                    return int(h)*3600 + int(m_)*60 + int(s) + int(cs)/100
                start_sec = ass_time_to_sec(start_str)
                end_sec = ass_time_to_sec(end_str)
                det = get_detections(video_path, start_sec, end_sec)
                return (_norm_key(start_sec,end_sec), det)
        return None
    with open(ass_path,'r',encoding="utf-8") as f:
        input_ass_file = f.read()
        log.info(f"input_ass_file:\n{input_ass_file}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(process_sub,line):i
            for i,line in enumerate(input_ass_file.splitlines())
            }
        results = {}
        for future in as_completed(future_to_idx):
            i= future_to_idx[future]
            try:
                result = future.result()
                log.info(f"original [{i}] : {input_ass_file.splitlines()[i]}")
                log.info(f"result [{i}] : {result}")
                if result:
                    key, det = result
                    DETECTION_CACHE[key] = det
                    results[i] = det
                log.info(f"completed[{i}]")
            except Exception as e:
                log.error(f"Error processing line '{input_ass_file.splitlines()[i]}',Error:{e}")
                traceback.print_exc()
    return results

def detect_text_ssa(video_path, ssa_path,  max_workers =5 ):
    """Parse SSA dialogue, compute detections and populate DETECTION_CACHE."""
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
                det = get_detections(video_path, start_sec, end_sec)
                return (_norm_key(start_sec,end_sec), det)
        return None

    with open(ssa_path, 'r', encoding="utf-8") as f:
        input_ssa_file = f.read()
        log.info(f"input_ssa_file:\n{input_ssa_file}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_as_idx =  { executor.submit(process_sub,line):i for i,line in enumerate(input_ssa_file.splitlines())}
        results = {}
        k=0
        for future in as_completed(future_as_idx):
            i = future_as_idx[future]
            try:
                result = future.result()
                log.info(f"original[{i}] : {input_ssa_file.splitlines()[i]}")
                log.info(f"result [{i}] : {result}")
                if result:
                    key, det = result
                    DETECTION_CACHE[key] = det
                    results[k] = det
                    k+=1
            except:
                log.error(f"Error processing line: {input_ssa_file.splitlines()[i]}")
    return results

def detect_text_vtt(video_path, vtt_path, max_workers = 5):
    """Parse VTT cues, compute detections and populate DETECTION_CACHE."""
    log.info("detect_text_vtt starting")
    min_frames = 3
    def vtt_time_to_sec(ts: str) -> float:
        hms = ts.strip().split(":")
        if len(hms) == 3:  # hh:mm:ss.mmm
            h, m, s_ms = hms
        else:  # mm:ss.mmm
            h, m, s_ms = 0, *hms
        s, ms = s_ms.split(".")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
    def extract_timestamps_string(line):
        matches = re.search(r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{1,3})",line)
        start_str = matches.group(1)
        end_str = matches.group(2)
        return start_str , end_str
    def process_sub(line: str):
        if "-->" in line:
            start_str , end_str = extract_timestamps_string(line)
            start_sec = vtt_time_to_sec(start_str)
            end_sec = vtt_time_to_sec(end_str)
            det = get_detections(video_path,start_sec,end_sec,min_frames)
            return (_norm_key(start_sec,end_sec), det)
        return None

    with open(vtt_path, "r", encoding="utf-8") as f:
        input_vtt_file = f.read()
        log.info(f"input_vtt_file:\n{input_vtt_file}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_as_idx = {executor.submit(process_sub,line):i for i,line in enumerate(input_vtt_file.splitlines())}
        results = {}
        for future in as_completed(future_as_idx):
            i = future_as_idx[future]
            try:
                result = future.result()
                log.info(f"original [{i}] : {input_vtt_file.splitlines()[i]}")
                log.info(f"result  [{i}] : {result}")
                if result:
                    key, det = result
                    DETECTION_CACHE[key] = det
                    results[i] = det
            except Exception as e:
                log.error(f"Error processing line: {input_vtt_file.splitlines()[i]} ({e})")
                log.error("error occured:",exc_info=True)
    return results

def display_results(results):
    results = dict(sorted(results.items()))
    for index , result in results.items():
        log.info(f"index:{index}")
        log.info(f"result:{result}")

import time
def process_subtitle(video_path, subtitle_path, max_workers = 12):
    """High-level process: first run detect_text_* to populate DETECTION_CACHE, then reposition_* reusing cached results."""
    start = time.time()
    ext = os.path.splitext(subtitle_path)[1].lower()

    # First phase: run detection and populate cache (get_text_* equivalent)
    if ext == ".srt":
        results = detect_text_srt(video_path,subtitle_path,max_workers=max_workers)
        if results:
            display_results(results)
        else:
            print("No results")
        # Second phase: use cache to generate output without re-running OCR
        output_file = os.path.splitext(subtitle_path)[0] + "_repositioned.ass"
        reposition_srt(video_path, subtitle_path, output_file ,max_workers=max_workers)
    elif ext ==".ass":
        results = detect_text_ass(video_path,subtitle_path,max_workers=max_workers)
        if results:
            display_results(results)
        else:
            print("No results")
        output_file = os.path.splitext(subtitle_path)[0] + "_repositioned.ass"
        reposition_ass(video_path, subtitle_path, output_file, max_workers=max_workers)
    elif ext == ".ssa":
        results = detect_text_ssa(video_path,subtitle_path,max_workers=max_workers)
        if results:
            display_results(results)
        else:
            print("No results")
        output_file = os.path.splitext(subtitle_path)[0] + "_repositioned.ssa"
        reposition_ssa(video_path, subtitle_path, output_file, max_workers=max_workers)
    elif ext == ".vtt":
        results = detect_text_vtt(video_path,subtitle_path,max_workers=max_workers)
        if results:
            display_results(results)
        else:
            print("No results")
        output_file = os.path.splitext(subtitle_path)[0] + "_repositioned.vtt"
        reposition_vtt(video_path, subtitle_path, output_file, max_workers=max_workers)
    else:
        raise ValueError(f"Unsupported subtitle format: {ext}")

    log.info(f"Repositioned subtitle saved: {output_file}")
    print(f"Repositioned subtitle saved: {output_file}")
    end = time.time()
    print("total time taken",end-start)
    return output_file

if __name__ == "__main__":
    # Example usage (adjust paths as needed)
    video_path = r"uploads\output.mp4"
    sub_path = r"outputs\Key_and_Peele_sample1.vtt"  # Try .srt, .ass, or .ssa too
    max_workers = 10
    process_subtitle(video_path, sub_path , max_workers)

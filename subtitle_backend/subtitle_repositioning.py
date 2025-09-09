import os

import pysrt
import re
# from rapidocr_test import detect_using_rapidocr
from rapidocr_onnxruntime import RapidOCR # for detection using rapidocr_onnxruntime
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
a=0
# def detect_using_rapidocr(img):
#     global a
#     print(f"a:{a}")
#     log.info(f"a:{a}")
#     result = engine(img)
#     # log.info(f"result:{result}")
#     results = result.to_json()
#     if results:
#         log.info("detections found for frame")
#         for i,temp_result in enumerate(results):
#             print("=====================================")
#             print(i,'\n',temp_result)
#             print("======================================")
#         result.vis(rf"rapid_ocr_frames\result_{a}.jpg")
#     else:
#         log.info("no detections found for frame")
#         cv2.imwrite(rf"rapid_ocr_frames\result_{a}.jpg",img)
    
#     a+=1
#     return results
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

def preprocess_threshold(image):
    """Prepare image for OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return thresh
import cv2
import numpy as np

def preprocess_using_blackout(image, block_size=50, contrast_thresh=0.3, blur_thresh=100.0):
    """
    Preprocess image by blacking out low-contrast or blurry regions.

    Args:
        image (numpy.ndarray): OpenCV-loaded image.
        block_size (int): Size of the region to check (pixels).
        contrast_thresh (float): Contrast threshold (lower = blacked out).
        blur_thresh (float): Laplacian variance threshold for blur.

    Returns:
        numpy.ndarray: Processed image with blacked-out regions.
    """
    output = image.copy()
    h, w = image.shape[:2]

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            roi = image[y:y+block_size, x:x+block_size]

            if roi.size == 0:
                continue

            # Contrast check
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            I_max, I_min = np.max(gray), np.min(gray)
            contrast = (I_max - I_min) / (I_max + I_min + 1e-5)

            # Blur check
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Blackout if low quality
            if contrast < contrast_thresh or variance < blur_thresh:
                output[y:y+block_size, x:x+block_size] = (0, 0, 0)

    return output
def preprocess(image, block_size=50, contrast_thresh=0.3, blur_thresh=100.0,
               adaptive_block=15, adaptive_C=5):
    """
    Preprocess image by:
    1. Blacking out low-contrast or blurry regions.
    2. Applying adaptive thresholding.

    Args:
        image (numpy.ndarray): OpenCV-loaded image.
        block_size (int): Size of the region to check (pixels).
        contrast_thresh (float): Contrast threshold (lower = blacked out).
        blur_thresh (float): Laplacian variance threshold for blur.
        adaptive_block (int): Block size for adaptive threshold (must be odd).
        adaptive_C (int): Constant subtracted in adaptive threshold.

    Returns:
        numpy.ndarray: Processed binary image (after blackouts + thresholding).
    """
    output = image.copy()
    h, w = image.shape[:2]

    # Step 1: Blackout low-quality regions
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            roi = image[y:y+block_size, x:x+block_size]
            if roi.size == 0:
                continue

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            I_max, I_min = np.max(gray), np.min(gray)
            contrast = (I_max - I_min) / (I_max + I_min + 1e-5)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()

            if contrast < contrast_thresh or variance < blur_thresh:
                output[y:y+block_size, x:x+block_size] = (0, 0, 0)

    # Step 2: Adaptive threshold
    gray_full = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray_full, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # or cv2.ADAPTIVE_THRESH_MEAN_C
        cv2.THRESH_BINARY,
        adaptive_block,
        adaptive_C
    )

    return binary

def preprocess_adaptive_threshold(image):
    """Prepare image for OCR using adaptive threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # or cv2.ADAPTIVE_THRESH_MEAN_C
        cv2.THRESH_BINARY, 
        blockSize=15,   # size of neighbourhood area (must be odd)
        C=5             # constant subtracted from mean
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
    log.info("in get position for segment")
    log.info(f"start sec:{start_sec},end_sec:{end_sec}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    #---------------------dynamically assign min frames-----------------------------------
    # total_frames_in_time_window = end_frame-start_frame+1
    sub_time = end_sec-start_sec
    # required_min_frames = int((0.05)*total_frames_in_time_window)
    required_min_frames = int(sub_time)//2
    min_frames = max(min_frames,required_min_frames)
    log.info(f"required min frames is {required_min_frames}")
    log.info(f"min frames now {min_frames}")
    #---------------------dynamically assign min frames-----------------------------------
    frame_indices = np.linspace(start_frame, end_frame, min(min_frames, abs(end_frame - start_frame + 1)), dtype=int)

    filtered_detections_per_frame = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            log.info("could not obtain frame")
            continue
        # preprocessed = preprocess_threshold(frame)
        preprocessed = preprocess_adaptive_threshold(frame)
        # preprocessed = preprocess_using_blackout(frame)
        # preprocessed = preprocess(frame)
        # preprocessed = frame
        detections = detect_using_rapidocr(preprocessed)
        filtered_detections_per_frame.append(detections)
    log.info(f"filtered detections per frame{filtered_detections_per_frame}")
    cap.release()

    return decide_subtitle_position(filtered_detections_per_frame, frame_height)

from concurrent.futures import ThreadPoolExecutor, as_completed

def reposition_srt(video_path, srt_path, output_ass_path,results , min_frames=3, max_workers=5):
    """Read SRT, run OCR in parallel (5 lines at a time), output ASS with repositioned alignment tags."""
    subs = pysrt.open(srt_path)

    def process_sub(sub_index,position):
        """Process one subtitle line: run OCR on segment and decide position."""
        sub = next((s for s in subs if s.index == sub_index), None)
        log.info("sub obtained")
        log.info(f"sub text:{sub.text}")
        alignment_tag = r"{\an8}" if position == "top" else r"{\an2}"
        formatted_text = sub.text.replace("\n", r"\N")
        line = (
            f"Dialogue: 0,{to_ass_timestamp(sub.start)},{to_ass_timestamp(sub.end)},"
            f"Default,,0,0,0,,{alignment_tag}{formatted_text}\n"
        )
        return line

    # Run OCR for multiple subtitles in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(process_sub,results[result].get('subtitle_index'),results[result].get('recommended_position')): i for i, result in enumerate(results)}
        results = {}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                log.error(f"Error processing subtitle {idx}: {e}")
                results[idx] = None

    # Write results back in correct order
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
            if results[i]:
                f.write(results[i])

    log.info(f"Repositioned subtitle saved: {output_ass_path}")
    return output_ass_path

import traceback
def reposition_ass(ass_path, results, output_ass_path, max_workers = 5):
    """Modify only alignment tags in ASS dialogue lines using precomputed detection results.

    The 'results' should be a mapping where each value has:
      - 'subtitle_index': index into the lines array for the Dialogue line
      - 'recommended_position': 'top' or 'bottom'
    This function reuses the provided detection results and does not recompute detection.
    """
    with open(ass_path, 'r') as f:
        input_ass_lines = f.read().splitlines()
        log.info(f"input_ass_file:\n{input_ass_lines}")

    output_ass_lines = list(input_ass_lines)

    def process_by_index(sub_index: int, position: str) -> tuple[int, str]:
        """Apply alignment change for a single ASS Dialogue line given index and position."""
        if sub_index < 0 or sub_index >= len(input_ass_lines):
            return sub_index, ""
        line = input_ass_lines[sub_index]
        if line.startswith("Dialogue:"):
            # Replace or insert alignment tag strictly based on provided position
            if re.search(r"\{\\an\d\}", line):
                line = re.sub(r"\{\\an\d\}", r"{\\an8}" if position == "top" else r"{\\an2}", line)
            else:
                line = line.rstrip("\n") + (r"{\an8}" if position == "top" else r"{\an2}")
        return sub_index, line

    # Launch updates using provided results only
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                process_by_index,
                results[key].get('subtitle_index'),
                results[key].get('recommended_position')
            ): key
            for key in results
        }
        for fut in as_completed(future_to_idx):
            try:
                idx, new_line = fut.result()
                if new_line:
                    output_ass_lines[idx] = new_line
            except Exception as e:
                log.error("Error updating ASS line for key %s: %s", future_to_idx[fut], e, exc_info=True)

    new_ass_file = "\n".join(output_ass_lines) if output_ass_lines else ""
    log.info(new_ass_file)
    with open(output_ass_path, "w", encoding="utf-8") as fo:
        fo.write(new_ass_file)

def reposition_ssa(ssa_path, results, output_ssa_path , max_workers =5 ):
    """Modify only alignment tags in SSA dialogue lines using precomputed detection results.

    The 'results' should be a mapping where each value has:
      - 'subtitle_index': index into the lines array for the Dialogue line
      - 'recommended_position': 'top' or 'bottom'
    This function reuses the provided detection results and does not recompute detection.
    """
    with open(ssa_path, 'r', encoding="utf-8") as f:
        input_ssa_lines = f.read().splitlines()
        log.info(f"input_ssa_file:\n{input_ssa_lines}")
    output_ssa_lines = list(input_ssa_lines)

    def process_by_index(sub_index: int, position: str) -> tuple[int, str]:
        """Apply alignment tag update on SSA Dialogue line by provided index and position."""
        if sub_index < 0 or sub_index >= len(input_ssa_lines):
            return sub_index, ""
        line = input_ssa_lines[sub_index]
        if line.startswith("Dialogue:"):
            if re.search(r"\{\\an\d\}", line):
                line = re.sub(r"\{\\an\d\}", r"{\\an8}" if position == "top" else r"{\\an2}", line)
            else:
                line = line.rstrip("\n") + (r"{\an8}" if position == "top" else r"{\an2}")
        return sub_index, line

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {
            executor.submit(
                process_by_index,
                results[key].get('subtitle_index'),
                results[key].get('recommended_position')
            ): key
            for key in results
        }
        for fut in as_completed(future_to_key):
            try:
                idx, updated_line = fut.result()
                if updated_line:
                    output_ssa_lines[idx] = updated_line
            except Exception as e:
                log.error("Error updating SSA line for key %s: %s", future_to_key[fut], e, exc_info=True)

    new_ssa_file = "\n".join(output_ssa_lines) if output_ssa_lines else ""
    log.info(new_ssa_file)
    with open(output_ssa_path, "w", encoding="utf-8") as f:
        f.write(new_ssa_file)

def reposition_vtt(vtt_path, results, output_vtt_path , max_workers = 5):
    """For VTT: modify 'line:' cue position using precomputed detection results.

    The 'results' should be a mapping where each value has:
      - 'subtitle_index': index into the lines array for the cue timing line
      - 'recommended_position': 'top' or 'bottom'
    This function reuses the provided detection results and does not recompute detection.
    """
    log.info("repositioning vtt file")

    with open(vtt_path, "r", encoding="utf-8") as f:
        input_vtt_lines = f.read().splitlines()
        log.info(f"input_vtt_file:\n{input_vtt_lines}")

    output_vtt_lines = list(input_vtt_lines)

    def process_by_index(sub_index: int, position: str) -> tuple[int, str]:
        """Apply 'line:' cue position update based on provided detection for a given index."""
        if sub_index < 0 or sub_index >= len(input_vtt_lines):
            return sub_index, ""
        line = input_vtt_lines[sub_index]
        if "-->" in line:
            if "line:" in line:
                line = re.sub(r"line:\d+%?", "line:0%" if position == "top" else "line:80%", line)
            else:
                line = line.strip() + (" line:0%" if position == "top" else " line:80%")
        return sub_index, line

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {
            executor.submit(
                process_by_index,
                results[key].get('subtitle_index'),
                results[key].get('recommended_position')
            ): key
            for key in results
        }
        for fut in as_completed(future_to_key):
            try:
                idx, new_line = fut.result()
                if new_line:
                    output_vtt_lines[idx] = new_line
            except Exception as e:
                log.error("Error updating VTT line for key %s: %s", future_to_key[fut], e, exc_info=True)

    new_vtt_file = "\n".join(output_vtt_lines) if output_vtt_lines else ""
    log.info(new_vtt_file)
    with open(output_vtt_path, "w", encoding="utf-8") as f:
        f.write(new_vtt_file)

#----------------------------------------------------printing out detections---------------------------------------------------

def get_detections(video_path, start_sec, end_sec, min_frames=3):
    """Run OCR on sampled frames to decide top/bottom."""
    log.info("in get position for segment")
    log.info(f"start sec:{start_sec},end_sec:{end_sec}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))       
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    #---------------------dynamically assign min frames-----------------------------------
    sub_time = end_sec-start_sec
    required_min_frames = int(sub_time)//2
    min_frames = max(min_frames,required_min_frames)
    #---------------------dynamically assign min frames-----------------------------------
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
    d = {"analysis":analysis,"recommended_position":recommended_position}
    return d

def detect_text_srt(video_path, srt_path, min_frames=3, max_workers=5):
    """Read SRT, run OCR in parallel (5 lines at a time), output ASS with repositioned alignment tags."""
    subs = pysrt.open(srt_path)

    def process_sub(sub,sub_index):
        """Process one subtitle line: run OCR on segment and decide position."""
        start_sec = sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds + sub.start.milliseconds / 1000
        end_sec = sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds + sub.end.milliseconds / 1000
        log.info(f"sub text:{sub.text}")
        detections = get_detections(video_path,start_sec,end_sec,min_frames)
        detections["subtitle_index"]=sub_index
        log.info(f"detections{detections}")
        return detections

    # Run OCR for multiple subtitles in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(process_sub, sub ,sub.index): i for i, sub in enumerate(subs)}
        results = {}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:                
                results[idx] = future.result()
                log.info("retrieved result")
            except Exception as e:
                log.error(f"Error processing subtitle {idx}: {e}")
                results[idx] = None
    return results



# import traceback
def detect_text_ass(video_path, ass_path, max_workers = 5):
    """Modify only alignment tags in ASS/SSA dialogue lines."""
    def process_sub(line:str ,sub_index):
            if line.startswith("Dialogue:"):
                # Parse times from Dialogue line
                m = re.match(r"Dialogue: \d+,(.*?),(.*?),", line)
                if m:
                    # log.info("found dialogue")
                    start_str, end_str = m.groups()
                    # Convert ASS time to seconds
                    def ass_time_to_sec(ts):
                        h, m_, s_cs = ts.split(":")
                        s, cs = s_cs.split(".")
                        return int(h)*3600 + int(m_)*60 + int(s) + int(cs)/100
                    start_sec = ass_time_to_sec(start_str)
                    end_sec = ass_time_to_sec(end_str)
                    # print(line)
                    matches = re.search(r"Dialogue: \d+,[0-9]:[0-9]{2}:[0-9]{2}.\d+,[0-9]{1,2}:[0-9]{2}:[0-9]{2}.\d+,(?:Default)?,(?:.*)?,\d+,\d+,\d+,(?:.*)?,(?:\{\\an\d\})?(.*)",line)
                    sub_text = matches.groups()[0]
                    log.info(f"sub text:{sub_text}")
                    detections = get_detections(video_path, start_sec, end_sec)
                    # detections["subtitle_index"] = s
                    return detections
                else:
                    return None
    with open(ass_path,'r') as f:
        input_ass_file = f.read()
        log.info(f"input_ass_file:\n{input_ass_file}")
    # output_ass_file = input_ass_file.splitlines()
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
                # if result:
                log.info(f"original [{i}] : {input_ass_file.splitlines()[i]}")
                log.info(f"result [{i}] : {result}")
                if result:
                    results[i] = result
                log.info(f"completed[{i}]")
                # results.append(result)
            except Exception as e:
                log.error(f"Error processing line '{input_ass_file.splitlines()[i]}',Error:{e}")
                traceback.print_exc()
    return results

        
def detect_text_ssa(video_path, ssa_path,  max_workers =5 ):
    """Modify only alignment tags in SSA dialogue lines."""

    def process_sub(line: str):
        if line.startswith("Dialogue:"):
            # Parse times from Dialogue line
            m = re.match(r"Dialogue: Marked=\d+,(.*?),(.*?),", line)
            if m:
                start_str, end_str = m.groups()

                # Convert SSA time to seconds
                def ssa_time_to_sec(ts):
                    h, m_, s_cs = ts.split(":")
                    s, cs = s_cs.split(".")
                    return int(h) * 3600 + int(m_) * 60 + int(s) + int(cs) / 100

                start_sec = ssa_time_to_sec(start_str)
                end_sec = ssa_time_to_sec(end_str)

                # Capture subtitle text (with optional alignment tag)
                matches = re.search(
                    r"Dialogue: Marked=\d+,[0-9]:[0-9]{2}:[0-9]{2}\.\d+,[0-9]{1,2}:[0-9]{2}:[0-9]{2}\.\d+,(?:Default)?,(?:.*)?,\d+,\d+,\d+,,(?:\{\\an\d\})?(.*)",
                    line
                )
                if matches:
                    sub_text = matches.groups()[0]
                    log.info(f"sub text: {sub_text}")

                    detections = get_detections(video_path, start_sec, end_sec)
                    return detections
                else:
                    return None
        else:
            return None

    with open(ssa_path, 'r', encoding="utf-8") as f:
        input_ssa_file = f.read()
        log.info(f"input_ssa_file:\n{input_ssa_file}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # futures = [executor.submit(process_sub, line) for line in input_ssa_file.splitlines()]
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
                    results[k] = result
                    k+=1
            except:
                log.error(f"Error processing line: {input_ssa_file.splitlines()[i]}")
    return results

def detect_text_vtt(video_path, vtt_path, max_workers = 5):
    """For VTT: modify 'line:' cue position like ASS/SSA alignment."""
    log.info("repositioning vtt file")
    min_frames = 3
    def vtt_time_to_sec(ts: str) -> float:
        # VTT timestamps are usually: hh:mm:ss.mmm or mm:ss.mmm
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
            log.info(f"start_str_new : {start_str} , end_str_new : {end_str}")
            start_sec = vtt_time_to_sec(start_str)
            end_sec = vtt_time_to_sec(end_str)
            detections = get_detections(video_path,start_sec,end_sec,min_frames) 
            return detections
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
                    results[i] = result
                 
            except Exception as e:
                log.error(f"Error processing line: {input_vtt_file.splitlines()[i]} ({e})")
                log.error("error occured:",exc_info=True)
    return results

#-----------------------------------------------------printing out detections----------------------------------------------
def display_results(results):
    results = dict(sorted(results.items()))
    for index , result in results.items():
        log.info(f"index:{index}")
        log.info(f"result:{result}")
import time
def process_subtitle(video_path, subtitle_path, max_workers = 12):
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
    elif ext ==".ass":
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

    log.info(f"Repositioned subtitle saved: {output_file}")
    print(f"Repositioned subtitle saved: {output_file}")
    end = time.time()
    print("total time taken",end-start)
    return output_file

if __name__ == "__main__":
    #---------------------15 min -----------------------------------------
    # video_path = r"..\videos_and_srt\Truck_Drivers_in_India.webm"
    # sub_path = r"..\videos_and_srt\Truck_Drivers.ass"  # Try .ass or .vtt too
    # process_subtitle(video_path, sub_path)
    #---------------------15 min -----------------------------------------
    #---------------------1/2 min -----------------------------------------
    video_path = r"uploads\Key_and_Peele_sample1.mp4"
    sub_path = r"outputs\Key_and_Peele_sample1.ass"  # Try .ass or .vtt too
    max_workers = 10
    process_subtitle(video_path, sub_path , max_workers)    
    #---------------------1/2 min -----------------------------------------


# videos_and_srt/The Seinfeld Chronicles.mp4
# videos_and_srt/MEDICAMENTOS 1.mp4
# videos_and_srt/Truck_Drivers_in_India.webm
    # resp = reposition_ass(r"../videos_and_srt/Key_and_Peele_sample1.mp4",r"../videos_and_srt/Key_and_Peele_sample1_repositioned.ass","sample_output_ass.ass")

"""
Centralized subtitle repositioning module.

This module exposes:
- process_subtitles(video_path, subtitle_path, max_workers=12)

It parses the subtitle file, precomputes detection 'results' (top/bottom decisions per segment),
and passes this 'results' mapping into the specific reposition functions which strictly reuse it.
No reposition function should perform detection or access results by other means.
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from typing import Dict, Iterable, List, Tuple

# Minimal imports; detection/precomputation assumed handled inside process_subtitles
# which will compute 'results' and pass it down.

# For SRT parsing
import srt


def _to_ass_timestamp_from_timedelta(td: timedelta) -> str:
    """Convert a timedelta to ASS H:MM:SS.CC format."""
    total_ms = int(td.total_seconds() * 1000)
    hours = total_ms // 3600000
    minutes = (total_ms % 3600000) // 60000
    seconds = (total_ms % 60000) // 1000
    centiseconds = (total_ms % 1000) // 10
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def _parse_srt_file(path: str) -> List[srt.Subtitle]:
    """Parse SRT file contents into a list of srt.Subtitle entries."""
    with open(path, "r", encoding="utf-8-sig") as f:
        contents = f.read()
    return list(srt.parse(contents))


def _ass_time_to_sec(ts: str) -> float:
    """Convert ASS timestamp 'H:MM:SS.CS' into seconds."""
    h, m_, s_cs = ts.split(":")
    s, cs = s_cs.split(".")
    return int(h) * 3600 + int(m_) * 60 + int(s) + int(cs) / 100


def _ssa_time_to_sec(ts: str) -> float:
    """Convert SSA timestamp 'H:MM:SS.CS' into seconds."""
    h, m_, s_cs = ts.split(":")
    s, cs = s_cs.split(".")
    return int(h) * 3600 + int(m_) * 60 + int(s) + int(cs) / 100


def _vtt_time_to_sec(ts: str) -> float:
    """Convert VTT timestamp into seconds (supports hh:mm:ss.mmm and mm:ss.mmm)."""
    hms = ts.strip().split(":")
    if len(hms) == 3:
        h, m, s_ms = hms
    else:
        h = 0
        m, s_ms = hms
    s, ms = s_ms.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


# PUBLIC_INTERFACE
def reposition_srt(srt_path: str, output_ass_path: str, results: Dict[Tuple[float, float], str], max_workers: int = 5) -> str:
    """Reposition SRT cues using provided results mapping and write an ASS file.

    The 'results' argument maps (start_sec, end_sec) -> 'top' or 'bottom'.
    This function uses ONLY the provided results and never re-computes detection.
    """
    subs = _parse_srt_file(srt_path)
    segments = [(float(sub.start.total_seconds()), float(sub.end.total_seconds())) for sub in subs]

    def position_for_segment(start_sec: float, end_sec: float) -> str:
        # Exact key first; then rounded to reduce float key mismatch
        return results.get((start_sec, end_sec)) or results.get((round(start_sec, 3), round(end_sec, 3))) or "bottom"

    def process_sub(i_sub: int):
        sub = subs[i_sub]
        start_sec, end_sec = segments[i_sub]
        position = position_for_segment(start_sec, end_sec)
        alignment_tag = r"{\an8}" if position == "top" else r"{\an2}"
        formatted_text = sub.content.replace("\n", r"\N")
        line = (
            f"Dialogue: 0,{_to_ass_timestamp_from_timedelta(sub.start)},{_to_ass_timestamp_from_timedelta(sub.end)},"
            f"Default,,0,0,0,,{alignment_tag}{formatted_text}\n"
        )
        return i_sub, line

    workers = max(1, min(int(max_workers), 12))
    lines_out = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_sub, i) for i in range(len(subs))]
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

    return output_ass_path


# PUBLIC_INTERFACE
def reposition_ass(ass_path: str, output_ass_path: str, results: Dict[Tuple[float, float], str], max_workers: int = 5) -> str:
    """Reposition ASS dialogue lines using provided results mapping.

    The 'results' argument maps (start_sec, end_sec) -> 'top' or 'bottom'.
    This function uses ONLY the provided results and never re-computes detection.
    """
    def position_for_segment(start_sec: float, end_sec: float) -> str:
        return results.get((start_sec, end_sec)) or results.get((round(start_sec, 3), round(end_sec, 3))) or "bottom"

    def process_line(line: str) -> str:
        if line.startswith("Dialogue:"):
            m = re.match(r"Dialogue: \d+,(.*?),(.*?),", line)
            if m:
                start_str, end_str = m.groups()
                start_sec = _ass_time_to_sec(start_str)
                end_sec = _ass_time_to_sec(end_str)
                position = position_for_segment(start_sec, end_sec)
                if re.search(r"\{\\an\d\}", line):
                    line = re.sub(r"\{\\an\d\}", r"{\an8}" if position == "top" else r"{\an2}", line)
                else:
                    line = line.rstrip("\n") + (r"{\an8}" if position == "top" else r"{\an2}") + "\n"
        return line

    with open(ass_path, "r", encoding="utf-8") as f:
        input_ass_file = f.read()

    output_lines = input_ass_file.splitlines(keepends=True)
    workers = max(1, min(int(max_workers), 12))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {executor.submit(process_line, line): i for i, line in enumerate(output_lines)}
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            output_lines[i] = future.result()

    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.writelines(output_lines)

    return output_ass_path


# PUBLIC_INTERFACE
def reposition_ssa(ssa_path: str, output_ssa_path: str, results: Dict[Tuple[float, float], str], max_workers: int = 5) -> str:
    """Reposition SSA dialogue lines using provided results mapping.

    The 'results' argument maps (start_sec, end_sec) -> 'top' or 'bottom'.
    This function uses ONLY the provided results and never re-computes detection.
    """
    def position_for_segment(start_sec: float, end_sec: float) -> str:
        return results.get((start_sec, end_sec)) or results.get((round(start_sec, 3), round(end_sec, 3))) or "bottom"

    def process_line(line: str) -> str:
        if line.startswith("Dialogue:"):
            m = re.match(r"Dialogue: Marked=\d+,(.*?),(.*?),", line)
            if m:
                start_str, end_str = m.groups()
                start_sec = _ssa_time_to_sec(start_str)
                end_sec = _ssa_time_to_sec(end_str)
                position = position_for_segment(start_sec, end_sec)
                if re.search(r"\{\\an\d\}", line):
                    line = re.sub(r"\{\\an\d\}", r"{\an8}" if position == "top" else r"{\an2}", line)
                else:
                    line = line.rstrip("\n") + (r"{\an8}" if position == "top" else r"{\an2}") + "\n"
        return line

    with open(ssa_path, "r", encoding="utf-8") as f:
        input_ssa_file = f.read()

    lines = input_ssa_file.splitlines(keepends=True)
    workers = max(1, min(int(max_workers), 12))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {executor.submit(process_line, line): i for i, line in enumerate(lines)}
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            lines[i] = future.result()

    with open(output_ssa_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return output_ssa_path


# PUBLIC_INTERFACE
def reposition_vtt(vtt_path: str, output_vtt_path: str, results: Dict[Tuple[float, float], str], max_workers: int = 5) -> str:
    """Reposition VTT cue 'line:' property using provided results mapping.

    The 'results' argument maps (start_sec, end_sec) -> 'top' or 'bottom'.
    This function uses ONLY the provided results and never re-computes detection.
    """
    ts_re = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{1,3})")

    def position_for_segment(start_sec: float, end_sec: float) -> str:
        return results.get((start_sec, end_sec)) or results.get((round(start_sec, 3), round(end_sec, 3))) or "bottom"

    def process_line(line: str) -> str:
        if "-->" in line:
            m = ts_re.search(line)
            if m:
                start_str, end_str = m.groups()
                start_sec = _vtt_time_to_sec(start_str)
                end_sec = _vtt_time_to_sec(end_str)
                position = position_for_segment(start_sec, end_sec)
                if "line:" in line:
                    # replace existing line position
                    line = re.sub(r"line:\d+%?", "line:0%" if position == "top" else "line:80%", line)
                else:
                    # append line position
                    line = line.rstrip("\n") + (" line:0%" if position == "top" else " line:80%") + "\n"
        return line

    with open(vtt_path, "r", encoding="utf-8") as f:
        input_vtt_file = f.read()

    lines = input_vtt_file.splitlines(keepends=True)
    workers = max(1, min(int(max_workers), 12))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {executor.submit(process_line, line): i for i, line in enumerate(lines)}
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            lines[i] = future.result()

    with open(output_vtt_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return output_vtt_path


def _extract_segments_from_ass(ass_path: str) -> List[Tuple[float, float]]:
    """Extract (start_sec, end_sec) pairs from ASS Dialogue lines."""
    segments: List[Tuple[float, float]] = []
    with open(ass_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Dialogue:"):
                m = re.match(r"Dialogue: \d+,(.*?),(.*?),", line)
                if m:
                    start_str, end_str = m.groups()
                    segments.append((_ass_time_to_sec(start_str), _ass_time_to_sec(end_str)))
    return segments


def _extract_segments_from_ssa(ssa_path: str) -> List[Tuple[float, float]]:
    """Extract (start_sec, end_sec) pairs from SSA Dialogue lines."""
    segments: List[Tuple[float, float]] = []
    with open(ssa_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Dialogue:"):
                m = re.match(r"Dialogue: Marked=\d+,(.*?),(.*?),", line)
                if m:
                    start_str, end_str = m.groups()
                    segments.append((_ssa_time_to_sec(start_str), _ssa_time_to_sec(end_str)))
    return segments


def _extract_segments_from_vtt(vtt_path: str) -> List[Tuple[float, float]]:
    """Extract (start_sec, end_sec) pairs from VTT cue timing lines."""
    segments: List[Tuple[float, float]] = []
    ts_re = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{1,3})")
    with open(vtt_path, "r", encoding="utf-8") as f:
        for line in f:
            if "-->" in line:
                m = ts_re.search(line)
                if m:
                    start_str, end_str = m.groups()
                    segments.append((_vtt_time_to_sec(start_str), _vtt_time_to_sec(end_str)))
    return segments


def _compute_results_for_segments(segments: Iterable[Tuple[float, float]]) -> Dict[Tuple[float, float], str]:
    """
    Stub for detection results computation.

    Note: In this task, we assume 'process_subtitles' has already generated the 'results' mapping
    using external detection logic. If this module needs to simulate that, replace the logic below.

    For now, this raises NotImplementedError to signal callers must supply real detections here
    (the higher-level application should inject the actual results).
    """
    raise NotImplementedError("results computation must be provided by the caller/context")


# PUBLIC_INTERFACE
def process_subtitles(video_path: str, subtitle_path: str, max_workers: int = 12) -> str:
    """Process a subtitle file (srt/ass/ssa/vtt) and write repositioned output, returning the file path.

    This function must precompute the detection 'results' and pass them as an argument to the
    format-specific reposition function. The reposition functions must exclusively reuse 'results'
    and must not perform detection or consult any other source.
    """
    ext = os.path.splitext(subtitle_path)[1].lower()

    # Build segments set depending on type
    if ext == ".srt":
        subs = _parse_srt_file(subtitle_path)
        segments = [(float(sub.start.total_seconds()), float(sub.end.total_seconds())) for sub in subs]
        # Compute results once (caller integration should inject real computation)
        results = _compute_results_for_segments(segments)
        output_file = os.path.splitext(subtitle_path)[0] + "_repositioned.ass"
        reposition_srt(srt_path=subtitle_path, output_ass_path=output_file, results=results, max_workers=max_workers)
    elif ext == ".ass":
        segments = _extract_segments_from_ass(subtitle_path)
        results = _compute_results_for_segments(segments)
        output_file = os.path.splitext(subtitle_path)[0] + "_repositioned.ass"
        reposition_ass(ass_path=subtitle_path, output_ass_path=output_file, results=results, max_workers=max_workers)
    elif ext == ".ssa":
        segments = _extract_segments_from_ssa(subtitle_path)
        results = _compute_results_for_segments(segments)
        output_file = os.path.splitext(subtitle_path)[0] + "_repositioned.ssa"
        reposition_ssa(ssa_path=subtitle_path, output_ssa_path=output_file, results=results, max_workers=max_workers)
    elif ext == ".vtt":
        segments = _extract_segments_from_vtt(subtitle_path)
        results = _compute_results_for_segments(segments)
        output_file = os.path.splitext(subtitle_path)[0] + "_repositioned.vtt"
        reposition_vtt(vtt_path=subtitle_path, output_vtt_path=output_file, results=results, max_workers=max_workers)
    else:
        raise ValueError(f"Unsupported subtitle format: {ext}")

    return output_file

"""Streamlit presentation demo for the aerial vehicle outline project."""

from pathlib import Path

import cv2
import streamlit as st

from vehicle_detector import (
    CROP_XYXY,
    DEFAULT_PARAMS,
    DISPLAY_SCALE,
    draw_hough_segments,
    draw_overlay,
    load_crop,
    run_pipeline,
)


ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Aerial Vehicle Outline Demo",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.7rem; padding-bottom: 3rem; max-width: 1450px;}
      [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(22,31,49,.94), rgba(32,50,75,.94));
        border: 1px solid rgba(255,193,7,.26); border-radius: 14px; padding: 14px 18px;
      }
      [data-testid="stMetricLabel"] {color: #b9c4d2;}
      .hero-kicker {color:#f6b73c; letter-spacing:.13em; font-size:.78rem; font-weight:700;}
      .hero-copy {color:#aeb9c8; font-size:1.06rem; max-width:920px; margin-bottom:1.2rem;}
      .callout {border-left:4px solid #f6b73c; background:rgba(246,183,60,.08);
                padding:12px 16px; border-radius:0 10px 10px 0; margin:8px 0 18px;}
      .footer {color:#8995a5; font-size:.86rem; margin-top:2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_crop():
    return load_crop()


@st.cache_data(show_spinner="Grouping compatible Hough segments…")
def cached_pipeline(parameter_tuple):
    keys = (
        "canny_low",
        "canny_high",
        "closing_kernel",
        "hough_votes",
        "min_line_length",
        "max_line_gap",
    )
    params = dict(zip(keys, parameter_tuple))
    return run_pipeline(cached_crop(), params)


def to_rgb(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


st.markdown('<div class="hero-kicker">COMPUTER VISION • HOUGH TRANSFORM</div>', unsafe_allow_html=True)
st.title("Aerial vehicle outline detector")
st.markdown(
    '<div class="hero-copy">An explainable, classical vision pipeline that turns aerial pixels '
    'into approximate vehicle rectangles—without a trained model or hand-drawn boxes.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Live controls")
    st.caption("Move a control to rerun the detector. Defaults are the visually selected settings.")
    if st.button("Reset chosen settings", width="stretch"):
        for key, value in DEFAULT_PARAMS.items():
            st.session_state[key] = value
        st.rerun()

    canny_low = st.slider("Canny low threshold", 20, 110, DEFAULT_PARAMS["canny_low"], 5, key="canny_low")
    canny_high = st.slider("Canny high threshold", 80, 240, DEFAULT_PARAMS["canny_high"], 5, key="canny_high")
    closing_kernel = st.select_slider(
        "Closing kernel", options=[3, 5, 7, 9], value=DEFAULT_PARAMS["closing_kernel"], key="closing_kernel"
    )
    hough_votes = st.slider("Hough vote threshold", 15, 60, DEFAULT_PARAMS["hough_votes"], key="hough_votes")
    min_line_length = st.slider(
        "Minimum line length", 12, 60, DEFAULT_PARAMS["min_line_length"], key="min_line_length"
    )
    max_line_gap = st.slider("Maximum line gap", 4, 30, DEFAULT_PARAMS["max_line_gap"], key="max_line_gap")

    st.divider()
    st.markdown("**Chosen crop**")
    st.code(str(CROP_XYXY), language=None)
    st.caption(f"Coordinates in the 4032 × 3024 source; displayed at {DISPLAY_SCALE:g}×.")

if canny_low >= canny_high:
    st.error("Canny's low threshold must be smaller than its high threshold.")
    st.stop()

params = {
    "canny_low": canny_low,
    "canny_high": canny_high,
    "closing_kernel": closing_kernel,
    "hough_votes": hough_votes,
    "min_line_length": min_line_length,
    "max_line_gap": max_line_gap,
}
parameter_tuple = tuple(params[key] for key in DEFAULT_PARAMS)
crop_bgr = cached_crop()
pipeline = cached_pipeline(parameter_tuple)
overlay_bgr = draw_overlay(crop_bgr, pipeline["detections"])

metric_columns = st.columns(4)
metric_columns[0].metric("Final outlines", len(pipeline["detections"]))
metric_columns[1].metric("Raw Hough segments", f"{len(pipeline['lines']):,}")
metric_columns[2].metric("Compatible candidates", f"{pipeline['candidate_count']:,}")
metric_columns[3].metric("Crop resolution", f"{crop_bgr.shape[1]} × {crop_bgr.shape[0]}")

live_tab, pipeline_tab, study_tab, explain_tab = st.tabs(
    ["Live result", "Pipeline stages", "Parameter study", "Method & presentation"]
)

with live_tab:
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Input crop")
        st.image(to_rgb(crop_bgr), width="stretch")
    with right:
        st.subheader("Approximate vehicle outlines")
        st.image(to_rgb(overlay_bgr), width="stretch")
    st.markdown(
        '<div class="callout"><b>What to point out:</b> yellow boxes follow paired vehicle sides. '
        'Texture, local contrast, aspect ratio, border rejection, and overlap suppression remove most '
        'rectangles created by painted parking spaces.</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Current settings"):
        st.json(params)

with pipeline_tab:
    st.subheader("From image to outlines")
    row_one = st.columns(3)
    row_one[0].image(to_rgb(crop_bgr), caption="1 · Color crop", width="stretch")
    row_one[1].image(pipeline["gray"], caption="2 · Grayscale", width="stretch", clamp=True)
    row_one[2].image(pipeline["denoised"], caption="3 · Gaussian denoising", width="stretch", clamp=True)

    row_two = st.columns(3)
    row_two[0].image(pipeline["edges"], caption="4 · Canny edges", width="stretch", clamp=True)
    row_two[1].image(pipeline["closed"], caption="5 · Morphological closing", width="stretch", clamp=True)
    hough_bgr = draw_hough_segments(crop_bgr, pipeline["lines"])
    row_two[2].image(to_rgb(hough_bgr), caption="6 · Probabilistic Hough segments", width="stretch")

    st.image(to_rgb(overlay_bgr), caption="7 · Compatible segments grouped into approximate rectangles", width="stretch")

with study_tab:
    st.subheader("Why these defaults were selected")
    st.image(
        str(ROOT / "results" / "parameter_comparison.png"),
        caption="Six combinations of Canny, closing, and Hough parameters",
        width="stretch",
    )
    st.markdown(
        "**Setting C** balances edge continuity with clutter rejection. Lower thresholds retain pavement "
        "texture; larger closing kernels join markings and shadows; stricter Hough settings lose weaker "
        "vehicle sides. The notebook records every value used in this comparison."
    )

with explain_tab:
    st.subheader("60-second presentation flow")
    presentation_steps = [
        ("Problem", "Find vehicle-shaped rectangles in an oblique aerial image containing shadows and parking lines."),
        ("Edges", "Convert to grayscale, smooth noise, run Canny, then close short breaks."),
        ("Geometry", "Use probabilistic Hough segments and pair lines that are parallel, nearby, and overlapping."),
        ("Filtering", "Require vehicle-like size and aspect ratio, interior texture, local contrast, and low overlap."),
        ("Result", "Eight defensible approximate outlines remain with the selected parameters."),
    ]
    for number, (title, copy) in enumerate(presentation_steps, start=1):
        st.markdown(f"**{number}. {title}** — {copy}")

    st.subheader("Honest limitations")
    st.markdown(
        "- Boxes follow the strongest visible body or roof sides, so some do not cover bumper to bumper.\n"
        "- The adjacent red and dark vehicles are difficult to separate.\n"
        "- A dark vehicle left of center is missed because its body edge blends into its shadow.\n"
        "- This hand-designed detector is tuned to this scale and viewpoint; a learned detector would generalize better."
    )
    st.info("Use the sidebar during the demo: lower Canny thresholds to reveal clutter, then reset to show why the chosen setting is cleaner.")

st.markdown(
    '<div class="footer">Photograph: Heath Alseike, '
    '<a href="https://commons.wikimedia.org/wiki/File:Above_the_Parking_Lot_(24993869647).jpg">'
    'Above the Parking Lot</a> (2018), <a href="https://creativecommons.org/licenses/by/2.0/">'
    'CC BY 2.0</a>. Cropped, resized, processed, and annotated for this demonstration.</div>',
    unsafe_allow_html=True,
)

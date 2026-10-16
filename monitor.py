import os
import sys
import glob
import time
import shutil
import subprocess
import argparse
import cv2
from ultralytics import YOLO

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.detector import HotWorkComplianceAnalyzer
from src.logger import ViolationLogger

DEFAULT_MODEL_PATH = os.path.join("model", "hot_wotk (1).pt")
DEFAULT_DATA_DIR = "data"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

def parse_args():
    parser = argparse.ArgumentParser(description="Hot Work Safety Monitoring & Violation Detection System")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to trained YOLO model .pt file")
    parser.add_argument("--source", type=str, default=DEFAULT_DATA_DIR, help="Video/Image path, RTSP stream URL, directory path (e.g., 'data'), or webcam index (0)")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold (default 0.25)")
    parser.add_argument("--proximity", type=float, default=350.0, help="Danger zone radius in pixels (default 350)")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save annotated output video or image")
    parser.add_argument("--no-display", action="store_true", help="Run in headless mode without GUI window")
    parser.add_argument("--cooldown", type=float, default=3.0, help="Minimum seconds between violation snapshots")
    parser.add_argument("--max-disp-width", type=int, default=1280, help="Maximum preview window width (preserves aspect ratio)")
    return parser.parse_args()

def calculate_display_size(frame_width, frame_height, max_disp_width=1280):
    """Calculates preview window width and height while strictly preserving original aspect ratio."""
    if frame_width <= max_disp_width:
        return frame_width, frame_height
    scale = max_disp_width / float(frame_width)
    new_w = max_disp_width
    new_h = int(frame_height * scale)
    return new_w, new_h

def is_gui_available():
    """Checks if OpenCV HighGUI window operations are supported in the current environment."""
    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
        return False
    try:
        test_win = f"__test_gui_{int(time.time())}"
        cv2.namedWindow(test_win, cv2.WINDOW_NORMAL)
        cv2.destroyWindow(test_win)
        return True
    except (cv2.error, Exception):
        return False

def open_video_capture(source):
    """Attempts to open video capture with path resolution and backend fallbacks."""
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    if isinstance(source, str) and not str(source).isdigit():
        abs_source = os.path.abspath(source)
        if os.path.exists(abs_source):
            source = abs_source

    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        return cap, source

    if isinstance(source, str) and os.path.isfile(source):
        for backend in [cv2.CAP_FFMPEG, cv2.CAP_IMAGES, cv2.CAP_ANY]:
            try:
                cap = cv2.VideoCapture(source, backend)
                if cap.isOpened():
                    return cap, source
            except Exception:
                pass

    return cap, source

def get_video_writer(output_path, fps, width, height):
    """Tries optimal fourcc codecs for VideoWriter (mp4v -> XVID -> MJPG)."""
    codecs_to_try = ['mp4v', 'XVID', 'MJPG']
    for c in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*c)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if writer.isOpened():
                return writer
        except Exception:
            pass
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def ensure_browser_compatible_h264(output_path):
    """
    If ffmpeg CLI is available, re-encodes output video to standard H.264 (yuv420p)
    so it plays natively in web browsers, HTML5 video players, and Google Colab.
    """
    if not output_path or not os.path.exists(output_path):
        return

    ffmpeg_bin = shutil.which('ffmpeg')
    if not ffmpeg_bin:
        print("ℹ️ Note: ffmpeg CLI not found on system PATH. Output video saved with default OpenCV codec.")
        return

    temp_path = output_path + ".h264_temp.mp4"
    try:
        cmd = [
            ffmpeg_bin, "-y", "-i", output_path,
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-loglevel", "error", temp_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0 and os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            os.replace(temp_path, output_path)
            print(f"🎬 [H.264 ENCODED] Successfully re-encoded video for web/browser compatibility: {output_path}")
        else:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def process_single_image(img_path, model, analyzer, logger, args):
    """Processes a single image file for Hot Work compliance."""
    abs_img_path = os.path.abspath(img_path) if os.path.exists(img_path) else img_path
    print(f"\n🖼️ Processing Image: {abs_img_path}")

    if not os.path.exists(abs_img_path):
        print(f"❌ Image file not found: {abs_img_path}")
        return

    frame = cv2.imread(abs_img_path)
    if frame is None:
        print(f"❌ Failed to load image: {abs_img_path}")
        return

    results = model.predict(frame, conf=args.conf, verbose=False)
    analysis = analyzer.analyze(results)
    annotated_frame = logger.draw_annotations(frame, analysis)

    if analysis['has_violation']:
        logger.log_violation(frame, analysis, frame_id=1)
        print(f"🚨 [VIOLATION DETECTED] Image processed: {abs_img_path}")
    else:
        print(f"✅ [COMPLIANT / SAFE] No violations in: {abs_img_path}")

    out_path = args.output
    if not out_path:
        os.makedirs(os.path.join("logs", "annotated_images"), exist_ok=True)
        fname = os.path.basename(abs_img_path)
        out_path = os.path.join("logs", "annotated_images", f"result_{fname}")

    cv2.imwrite(out_path, annotated_frame)
    print(f"💾 Annotated image saved to: {out_path}")

    gui_available = not args.no_display and is_gui_available()
    if gui_available:
        try:
            win_title = f"Hot Work Compliance - {os.path.basename(abs_img_path)}"
            h, w = annotated_frame.shape[:2]
            disp_w, disp_h = calculate_display_size(w, h, args.max_disp_width)
            cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_title, disp_w, disp_h)
            cv2.imshow(win_title, annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except (cv2.error, Exception):
            pass

def process_video_source(source, model, analyzer, logger, args):
    """Processes a video file, RTSP stream, or webcam feed with aspect-ratio preserved display."""
    cap, resolved_source = open_video_capture(source)

    if not cap.isOpened():
        if isinstance(source, str) and not source.isdigit():
            if not os.path.exists(source):
                print(f"❌ Error: Video source file not found: '{source}'")
            else:
                size_mb = os.path.getsize(source) / (1024 * 1024)
                print(f"❌ Error: Could not decode video source '{source}' ({size_mb:.2f} MB). Verify video codecs or system OpenCV ffmpeg backend.")
        else:
            print(f"❌ Error: Could not open video source '{source}'")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if width <= 0 or height <= 0:
        width, height = 1280, 720
    disp_w, disp_h = calculate_display_size(width, height, args.max_disp_width)

    print(f"Input Stream Original Resolution: {width}x{height} @ {fps:.1f} FPS")

    writer = None
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        writer = get_video_writer(args.output, fps, width, height)
        print(f"Saving original full-res output video to: {args.output}")

    gui_available = not args.no_display and is_gui_available()
    win_title = "Hot Work Safety Compliance Monitor"
    if gui_available:
        try:
            cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_title, disp_w, disp_h)
            print(f"Preview Display Resolution:       {disp_w}x{disp_h} (Aspect Ratio Preserved)", flush=True)
        except (cv2.error, Exception):
            gui_available = False
            print("ℹ️ Headless environment detected (GUI display disabled). Running processing in background...", flush=True)
    else:
        print("ℹ️ Running in headless mode (GUI preview disabled).", flush=True)

    frame_count = 0
    start_time = time.time()

    print("\n🚀 Starting Video Monitoring...", flush=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream.")
                break

            frame_count += 1
            results = model.predict(frame, conf=args.conf, verbose=False)
            analysis = analyzer.analyze(results)
            annotated_frame = logger.draw_annotations(frame, analysis)
            logger.log_violation(frame, analysis, frame_id=frame_count)

            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0

            scale = max(0.6, min(width / 1280.0, height / 720.0))
            font_scale = max(0.5, 0.7 * scale)
            font_thick = max(1, int(2 * scale))
            cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (width - int(160 * scale), int(36 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick)

            if writer:
                writer.write(annotated_frame)

            if gui_available:
                try:
                    cv2.imshow(win_title, annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord('q'), 27]:
                        print("Monitoring stopped by user.")
                        break
                except (cv2.error, Exception):
                    gui_available = False

    except KeyboardInterrupt:
        print("\nMonitoring interrupted.")
    finally:
        cap.release()
        if writer:
            writer.release()
            ensure_browser_compatible_h264(args.output)

        if gui_available:
            try:
                cv2.destroyAllWindows()
            except (cv2.error, Exception):
                pass
        print(f"Processed {frame_count} frames.")

def main():
    args = parse_args()

    os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)

    print("=" * 65)
    print("🔥 HOT WORK SAFETY MONITORING SYSTEM")
    print("=" * 65)
    print(f"Model Path:          {args.model}")
    print(f"Target Source:       {args.source}")
    print(f"Confidence Threshold: {args.conf}")
    print(f"Proximity Radius:    {args.proximity} px")
    print(f"Data Directory:      {os.path.abspath(DEFAULT_DATA_DIR)}")
    print("=" * 65)

    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found at '{args.model}'")
        sys.exit(1)

    print("Loading YOLO Model...")
    model = YOLO(args.model)
    print("Model loaded successfully.")

    analyzer = HotWorkComplianceAnalyzer(proximity_threshold=args.proximity, conf_threshold=args.conf)
    logger = ViolationLogger(log_dir="logs", cooldown_seconds=args.cooldown)

    source = args.source

    if os.path.isdir(source):
        files = glob.glob(os.path.join(source, "*"))
        if not files:
            print(f"📁 Directory '{source}' is currently empty!")
            return

        print(f"📁 Found {len(files)} file(s) in directory '{source}':")
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                process_single_image(f, model, analyzer, logger, args)
            elif ext in VIDEO_EXTENSIONS:
                process_video_source(f, model, analyzer, logger, args)
            else:
                print(f"⚠️ Skipping unsupported file type: {f}")

    elif os.path.isfile(source) and os.path.splitext(source)[1].lower() in IMAGE_EXTENSIONS:
        process_single_image(source, model, analyzer, logger, args)

    else:
        process_video_source(source, model, analyzer, logger, args)

    print(f"\n✅ All tasks finished! Check your logs and snapshots in: {os.path.abspath('logs')}")

if __name__ == "__main__":
    main()


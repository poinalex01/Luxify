import time
from io import BytesIO
import os
import cv2
import numpy as np
from PIL import Image
from screeninfo import get_monitors
from playwright.sync_api import sync_playwright
import mediapipe as mp
from dotenv import load_dotenv
import base64

load_dotenv()
VDO_URL = os.getenv("VDO_URL")
if VDO_URL is None:
    raise RuntimeError(
        "VDO_URL environment variable not set. "
        "Please create a .env file with VDO_URL=<your_video_url> "
        "or set it in your system environment."
    )

TARGET_FPS = 60

# MediaPipe hand setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.4) as hands:
    with sync_playwright() as p:
        chromium_browser = p.chromium.launch(headless=True)
        monitor = get_monitors()[0]
        width, height = monitor.width // 2, monitor.height // 2
        page = chromium_browser.new_page(viewport={"width": width, "height": height})
        page.goto(VDO_URL)

        try:
            page.wait_for_selector("button", timeout=10000)
            for b in page.query_selector_all("button"):
                txt = (b.inner_text() or "").lower()
                if any(word in txt for word in ["start", "join", "enter", "ok", "allow"]):
                    b.click()
                    break
        except Exception:
            pass

        page.wait_for_selector("video", timeout=10000)
        video_el = page.query_selector("video")
        page.evaluate("(v)=>{v.muted=true; v.play().catch(()=>{});}", video_el)

        print("Press 'q' to quit.")
        frame_interval = 1.0 / TARGET_FPS

        # JS to capture video frame as base64
        js_capture = """
        (v)=>{
            const canvas = document.createElement('canvas');
            canvas.width = v.videoWidth;
            canvas.height = v.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(v, 0, 0, canvas.width, canvas.height);
            return canvas.toDataURL('image/png').split(',')[1];
        }
        """

        try:
            while True:
                t0 = time.time()

                # get frame from video using JS canvas
                img_base64 = page.evaluate(js_capture, video_el)
                img_bytes = BytesIO(base64.b64decode(img_base64))
                img = Image.open(img_bytes).convert("RGB")
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (width, height))

                # handtracking
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                cv2.imshow("Luxify", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                # fps throttle
                elapsed = time.time() - t0
                if (frame_interval - elapsed) > 0:
                    time.sleep(frame_interval - elapsed)

        finally:
            cv2.destroyAllWindows()
            chromium_browser.close()

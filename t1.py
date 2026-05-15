import json
import os
os.environ.pop("MKL_NUM_THREADS", None)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
import sys
sys.stdout.reconfigure(line_buffering=True)
import threading
import queue
import subprocess
import time
from collections import deque, defaultdict
from datetime import datetime
import base64
import traceback
import requests

import cv2
import numpy as np
import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
import torch.nn as nn
import torchvision.models as models
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Set
import uvicorn
import asyncio
from torchvision import transforms as T
from concurrent.futures import ThreadPoolExecutor
OCR_EXECUTOR = ThreadPoolExecutor(max_workers=2)


# ================= CONFIG =================
FRAME_WIDTH = 720
FRAME_HEIGHT = 1280

DISPLAY_WIDTH = 720
DISPLAY_HEIGHT = 1280
IMG_SIZE = 224
SEQ_LEN = 8
CLASSES = ["normal", "picked"]
CONF_THRESHOLD = 0.55
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.get_arch_list() and "sm_120" not in torch.cuda.get_arch_list() else "cpu")
WEBSOCKET_ENABLED = False
OCR_API_URL = "http://localhost:8082/api/pickup/event"


LOCAL_API = "http://localhost:8080/YARD/yardKalmar/"

SERVER_API = "http://localhost:8080/YARD/yardKalmar/"



BASE_DIR = r"E:\ocr\flocr\live code\kalmar-service"

LOG_DIR = os.path.join(BASE_DIR, "logs")
RECORD_BASE_DIR = os.path.join(BASE_DIR, "recordings")
OCR_IMAGE_BASE_DIR = os.path.join(BASE_DIR, "ocr_images")



BROADCAST_INTERVAL = 0.033
TARGET_MODEL_FPS = 6
MAX_OCR_ATTEMPTS = 10
OCR_INTERVAL = 12
ASYNC_LOOP = None
WAIT_BEFORE_TEMP = 0.1      # seconds after picked → normal
TEMP_PLACED_DURATION = 2.0   # how long temp placed is visible
MIN_OCR_WAIT_AFTER_FIRST = 6  # seconds (llama.cpp needs ~15s)
# ===== RECORDING CONFIG =====
MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB
DELAY_SEC = 1.0



# HEADLESS = os.environ.get("DISPLAY") is None
HEADLESS = False
RUN_AS_SERVICE = False


# ================= TESTING MODE CONFIG =================
# 🔧 CHANGE THIS TO True FOR VIDEO FILE TESTING
USE_VIDEO_FILE = True  # Set to False to use RTSP camera

# 🔧 PUT YOUR VIDEO FILE PATH HERE
VIDEO_FILE_PATH = r"E:\forklifter\4.mp4"  # Change this to your video path

# ================= SINGLE KALMAR CONFIG =================
SINGLE_KALMAR_ID = "A00003"
SINGLE_KALMAR = "KALMAR_1"

# Original RTSP URL (used when USE_VIDEO_FILE = False)
SINGLE_RTSP_URL = "rtsp://admin:Rapport%40123@192.168.1.212:554/video/live?channel=1&subtype=0"


# ===== RTSP RECONNECT CONFIG =====
RTSP_RECONNECT_DELAY = 5        # seconds
RTSP_MAX_FAILURES = 999999      # infinite retries
# ===== GPS CONFIG =====
DEVICE_ID = "123"
ECOCOSMO_API = f"https://abc.php?imei={DEVICE_ID}"
AUTH_TOKEN = "abcd"

LOCATION_INTERVAL_SEC = 2



print(f"🚀 Using device: {DEVICE}")
print(f"📹 Video mode: {'VIDEO FILE' if USE_VIDEO_FILE else 'RTSP CAMERA'}")
if USE_VIDEO_FILE:
    print(f"📁 Video path: {VIDEO_FILE_PATH}")


# ===== FSM STATE (from new model file) =====
current_state = "normal"
state_counter = 0
MIN_STATE_FRAMES = 6



# ================= GLOBAL STORES =================
# if WEBSOCKET_ENABLED:
clients: Dict[str, Set[WebSocket]] = {}
clients_lock = asyncio.Lock()   # ✅ CHANGE TO ASYNC LOCK
workers: Set[str] = set()


current_frames: Dict[str, np.ndarray] = {}
# GLOBAL display frames dictionary - main thread will read this
global_display_frames: Dict[str, np.ndarray] = {}
display_lock = threading.Lock()

kalmar_sessions = {}


MODEL_LOCK = threading.Lock()

# ============================================================
# GLOBAL SHUTDOWN EVENT
# ============================================================

shutdown_event = threading.Event()


def shutdown_application():

    print("\n🛑 SHUTTING DOWN APPLICATION...\n")

    shutdown_event.set()

    # Stop all kalmar sessions
    for session in list(kalmar_sessions.values()):

        try:
            session.stop()

        except Exception as e:
            print(f"Session stop error: {e}")

    # Close OpenCV windows
    try:
        cv2.destroyAllWindows()
    except:
        pass

    print("✅ All resources released")

    os._exit(0)


# Per-class confidence thresholds
CLASS_THRESHOLDS = {"normal": 0.45, "picked": 0.70}

transform = T.Compose(
    [
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)



# ============================================================
# PROFESSIONAL STATE DISPLAY HELPERS
# ============================================================


def draw_status_overlay(frame, state_name, confidence, kalmar_id=None, fps_display=0.0,
                        container_text=None, pickup_duration=None, ocr_in_progress=False):
    h, w = frame.shape[:2]

    colors = {
        "normal": (0, 200, 0),      # green
        "picked": (0, 140, 255),    # orange
        "placed": (255, 120, 0),    # blue-ish
        "ocr": (0, 255, 255),       # yellow
        "error": (0, 0, 255),       # red
    }

    color = colors.get(str(state_name).lower(), (255, 255, 255))

    # Dark translucent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (18, 18, 18), -1)
    frame = cv2.addWeighted(overlay, 0.60, frame, 0.40, 0)

    # State badge
    cv2.rectangle(frame, (15, 18), (175, 92), color, -1)
    cv2.putText(
        frame,
        str(state_name).upper(),
        (28, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Kalmar ID
    if kalmar_id:
        cv2.putText(
            frame,
            f"ID: {kalmar_id}",
            (195, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # Confidence pill
    cv2.rectangle(frame, (195, 58), (365, 96), (35, 35, 35), -1)
    cv2.putText(
        frame,
        f"CONF: {confidence:.2f}",
        (207, 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )

    # FPS pill
    cv2.rectangle(frame, (w - 155, 18), (w - 15, 54), (35, 35, 35), -1)
    cv2.putText(
        frame,
        f"FPS: {fps_display:.1f}",
        (w - 142, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Bottom info panel
    cv2.rectangle(frame, (15, 122), (w - 15, 185), (10, 10, 10), -1)

    if container_text:
        cv2.putText(
            frame,
            f"CONTAINER: {container_text}",
            (28, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )
    elif ocr_in_progress:
        cv2.putText(
            frame,
            "OCR Processing...",
            (28, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
    elif pickup_duration and pickup_duration > 0:
        cv2.putText(
            frame,
            f"Pickup: {pickup_duration:.1f}s",
            (28, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )

    return frame


# ============================================================
# LOCATION FETCHER (ECOSCOSMO PULLER)
# ============================================================

def monotonic_ms():
    return int(time.monotonic() * 1000)

# ================= AUX CAMERA SESSION =================

# aux_camera_sessions: Dict[str, AuxCameraSession] = {}



class KalmarLocationFetcher:
    def __init__(self, kalmar_id: str, ocr_handler):
        self.kalmar_id = kalmar_id
        self.ocr_handler = ocr_handler
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        if self.running:
            return
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        print("📍 Location polling started (Ecocosmo)")

    def _loop(self):
        headers = {"Authorization": AUTH_TOKEN}

        while self.running:
            try:
                r = requests.get(ECOCOSMO_API, headers=headers, timeout=10)

                if r.status_code == 200 and r.json():
                    loc = {
                        "latitude": r.json()[0]["latitude"],
                        "longitude": r.json()[0]["longitude"],
                        "timestamp": monotonic_ms(),  # ANDROID-style time
                    }

                    # ✅ THIS IS THE IMPORTANT LINE
                    self.ocr_handler.update_location(self.kalmar_id, loc)

                    # log_step(self.kalmar_id, "LOCATION_RECEIVED", loc)

            except Exception as e:
                print(f"[LOCATION] ❌ Fetch failed: {e}")

            time.sleep(LOCATION_INTERVAL_SEC)





class TemporalSmoother:
    def __init__(self, classes, window=8):
        self.window = window
        self.buffer = deque(maxlen=window)
        self.classes = classes
        self.last_change_time = time.time()
        self.last_smooth = None

    def update(self, pred, probs):
        self.buffer.append((pred, probs.copy()))

        if len(self.buffer) < self.window:
            return pred, float(probs[self.classes.index(pred)])

        score = {c: 0.0 for c in self.classes}
        count = {c: 0 for c in self.classes}

        for p, pr in self.buffer:
            count[p] += 1
            for i, c in enumerate(self.classes):
                score[c] += float(pr[i])

        best = max(score, key=score.get)
        conf = score[best] / len(self.buffer)

        if best != self.last_smooth:
            self.last_smooth = best
            self.last_change_time = time.time()

        return best, conf


class KalmarFSM:
    def __init__(self):
        self.state = "normal"
        self.enter_time = time.time()
        self.allowed = {
            "normal": ["picked"],
            "picked": ["normal"],   # ✅ FIX
            # "placed" is DERIVED, not a real FSM state
        }


    def can_transition(self, to_state):
        return to_state in self.allowed[self.state]

    def force(self, new_state):
        self.state = new_state
        self.enter_time = time.time()



kalmar_state = defaultdict(
    lambda: {
        "buffer": deque(maxlen=SEQ_LEN),
        "prev_action": "normal",
        "confidence": 0.0,
        "last_model_time": 0.0,
        "current_prediction": "",
        "smoother": TemporalSmoother(CLASSES, window=8),
        "fsm": KalmarFSM(),
        "fps_counter": deque(maxlen=30),
         # ✅ TRANSIENT PLACED STATE (NEW)
        "last_picked_time": None,
        "normal_since": None,
        "temp_placed_active": False,
        "temp_placed_until": 0.0,
        "temp_placed_emitted": False,   # prevents duplicates
        "placed_finalized": False,
        "placed_delay_started": False,
    }
)










# ================= LOCATION HANDLER =================
class LocationHandler:
    def __init__(self):
        self.location_history = {}  # kalmar_id -> deque of (timestamp, location_data)
        self.lock = threading.Lock()
        
    def update_location(self, kalmar_id: str, location_data: Dict):
        """Update location history for a kalmar"""
        with self.lock:
            if kalmar_id not in self.location_history:
                # Store last 100 locations (adjust as needed)
                self.location_history[kalmar_id] = deque(maxlen=300)
            
            # Store timestamp as float for comparison
            loc_timestamp = float(location_data.get("timestamp", time.time()))
            self.location_history[kalmar_id].append(
                (loc_timestamp, {
                    "latitude": location_data.get("latitude"),
                    "longitude": location_data.get("longitude"),
                    "timestamp": loc_timestamp,
                    "received_time": datetime.now().isoformat()
                })
            )
            
  

    def get_location_at_time(self, kalmar_id: str, target_timestamp: float) -> Optional[Dict]:
        """Get location with timestamp closest to target timestamp.
        If history is empty, return DEFAULT location.
        """

        DEFAULT_LOCATION = {
            "latitude": 18.949734385071864,
            "longitude": 73.15039574466999,
            "timestamp": target_timestamp,
            "received_time": datetime.now().isoformat(),
            "source": "default"
        }

        with self.lock:
            history = self.location_history.get(kalmar_id)

            print(
                f"[LOCATION_DEBUG] kalmar_id={kalmar_id} "
                f"history_size={len(history) if history else 0} "
                f"target_ts={target_timestamp}"
            )

            # ✅ FIX 1: history missing or empty → fallback
            if not history:
                print(
                    f"[LOCATION_FALLBACK] kalmar_id={kalmar_id} "
                    f"Using DEFAULT location"
                )

                log_step(kalmar_id, "LOCATION_FALLBACK_USED", {
                    "latitude": DEFAULT_LOCATION["latitude"],
                    "longitude": DEFAULT_LOCATION["longitude"],
                    "timestamp": target_timestamp
                })

                return DEFAULT_LOCATION

            # ✅ Normal case: find closest GPS point
            closest = min(history, key=lambda x: abs(x[0] - target_timestamp))
            return closest[1]



            
    def clear_history(self, kalmar_id: str):
        """Clear location history for a kalmar"""
        with self.lock:
            if kalmar_id in self.location_history:
                del self.location_history[kalmar_id]

    def dump_locations(self, kalmar_id: str):
        with self.lock:
            history = self.location_history.get(kalmar_id, [])
            print(f"\n📍 LOCATION DUMP for {kalmar_id} ({len(history)} points)")
            for i, (_, loc) in enumerate(history):
                print(
                    f"{i+1:03d} | "
                    f"lat={loc['latitude']} "
                    f"lon={loc['longitude']} "
                    f"ts={loc['timestamp']} "
                    f"recv={loc['received_time']}"
                )            


# ================= PLACED EVENT HANDLER =================
class PlacedEventHandler:
    def __init__(self):
        self.sent_events = set()

    def send_placed_event(self, kalmar_id: str, container_numbers: list, location: Dict, tracking_id :str):
        event_id = f"{kalmar_id}_{location['timestamp']}_{tracking_id}"

        if event_id in self.sent_events:
            return True

        payload = {
            "kalmar_id": kalmar_id,
            "container_numbers": container_numbers,  # ✅ ARRAY
            "event_type": "container_placed",
            "latitude": location.get("latitude"),
            "longitude": location.get("longitude"),
            "timestamp": location.get("timestamp"),
            "detected_time": datetime.now().isoformat(),
            "tracking_id": tracking_id
        }

        # -------------------------
        # 1️⃣ TRY MAIN SERVER FIRST
        # -------------------------
        try:
            r = requests.post(
                SERVER_API + "receivePlacedAutoEventPythonList",
                json=payload,
                timeout=5
            )

            if r.status_code == 200:
                self.sent_events.add(event_id)
                log_step(kalmar_id, "PLACED_API_MAIN_SUCCESS", payload)
                return True

            raise Exception(f"HTTP {r.status_code}")

        except Exception as e:
            log_step(kalmar_id, "PLACED_API_MAIN_FAILED", {
                "error": str(e)
            })

        # -------------------------
        # 2️⃣ FALLBACK → LOCAL TOMCAT
        # -------------------------
        try:
            r = requests.post(
                LOCAL_API + "receivePlacedAutoEventLocal",
                json=payload,
                timeout=3
            )

            if r.status_code == 200:
                log_step(kalmar_id, "PLACED_API_LOCAL_QUEUED", payload)
                return False

            log_step(kalmar_id, "PLACED_API_LOCAL_FAILED", {
                "status": r.status_code
            })

        except Exception as e:
            log_step(kalmar_id, "PLACED_API_LOCAL_ERROR", {
                "error": str(e)
            })

        return False

            
    def reset_for_kalmar(self, kalmar_id: str):
        """Reset sent events for a kalmar (when cycle restarts)"""
        to_remove = [eid for eid in self.sent_events if eid.startswith(f"{kalmar_id}_")]
        for eid in to_remove:
            self.sent_events.remove(eid)


# ================= OCR HANDLER =================
class OCRHandler:
    def __init__(self):
        self.pickup_timers = {}
        self.container_numbers = {}  # kalmar_id -> {container_no: count}
        self.container_lock = threading.Lock()
        self.max_ocr_attempts = MAX_OCR_ATTEMPTS
        self.ocr_interval = OCR_INTERVAL
        self.ocr_results = {}
        self.last_ocr_time = {}

        self.location_handler = LocationHandler()
        self.placed_event_handler = PlacedEventHandler()
        self.picked_snapshots = {}

    # ============================================================
    # PICKUP TRACKING (UNCHANGED LOGIC)
    # ============================================================

    def track_pickup(self, kalmar_id: str, action: str, confidence: float, frame):
        current_time = time.time()

        if action == "picked" and confidence >= CONF_THRESHOLD:

            if kalmar_id not in self.pickup_timers:
                self.picked_snapshots.pop(kalmar_id, None)

                self.pickup_timers[kalmar_id] = {
                    "start_time": current_time,
                    "stable_frames": 1,
                    "triggered": False,
                    "ocr_attempts": 0,
                    "container_found": False,
                    "last_ocr_sent": 0,
                    "stop": False   # ✅ ADD THIS
                }
            else:
                self.pickup_timers[kalmar_id]["stable_frames"] += 1

            timer = self.pickup_timers[kalmar_id]
            elapsed = current_time - timer["start_time"]

            if not timer["triggered"] and elapsed >= 5.0:
                timer["triggered"] = True
                
                tracking_id = tracking_manager.generate_tracking_id(SINGLE_KALMAR)
                kalmar_state[kalmar_id]["tracking_id"] = tracking_id

                log_step(kalmar_id, "TRACKING_ID_CREATED", {
                    "tracking_id": tracking_id
})


                OCR_EXECUTOR.submit(
                    self._start_continuous_ocr,
                    kalmar_id
                )

                self.location_handler.clear_history(kalmar_id)
                self.placed_event_handler.reset_for_kalmar(kalmar_id)

                # ✅ Capture single reference frame (unchanged behavior)
                live_frame = current_frames.get(kalmar_id)
                if live_frame is not None:
                    success, buffer = cv2.imencode(
                        ".jpg",
                        live_frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 100]
                    )
                    if success:
                        self.picked_snapshots[kalmar_id] = base64.b64encode(buffer).decode("utf-8")
                        log_step(kalmar_id, "PICKED_REFERENCE_CAPTURED")

                return True, elapsed

            return False, elapsed

        return False, 0.0

    # ============================================================
    # CONTINUOUS OCR (SINGLE LIVE FRAME ONLY)
    # ============================================================
    def _start_continuous_ocr(self, kalmar_id: str):
        attempt_count = 0
        log_step(kalmar_id, "OCR_THREAD_STARTED")

        #while True:
        while not shutdown_event.is_set():
            try:
                # ----------------------------
                # EXIT CONDITIONS
                # ----------------------------
                # ============================
            # ⛔ HARD STOP CHECK (ADD HERE)
            # ============================

                timer = self.pickup_timers.get(kalmar_id)

                if not timer:
                    log_step(kalmar_id, "OCR_STOP_NO_TIMER")
                    break

                if timer.get("stop"):
                    log_step(kalmar_id, "OCR_STOPPED_BY_PLACED")
                    break

                # if timer.get("container_found"):
                #     break

                if attempt_count >= self.max_ocr_attempts:
                    log_step(kalmar_id, "OCR_MAX_ATTEMPTS_REACHED")
                    break

                # ----------------------------
                # OCR INTERVAL CONTROL
                # ----------------------------
                if attempt_count > 0:
                    elapsed = time.time() - timer["last_ocr_sent"]
                    if elapsed < self.ocr_interval:
                        time.sleep(self.ocr_interval - elapsed)

                # ----------------------------
                # GET LIVE FRAME
                # ----------------------------
                live_frame = current_frames.get(kalmar_id)
                if live_frame is None:
                    time.sleep(0.1)
                    continue

                success, buffer = cv2.imencode(
                    ".jpg",
                    live_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 100]
                )

                if not success:
                    continue

                image_bytes = buffer.tobytes()
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")

                # ----------------------------
                # UPDATE ATTEMPT STATE
                # ----------------------------
                attempt_count += 1
                timer["ocr_attempts"] = attempt_count
                timer["last_ocr_sent"] = time.time()

                tracking_id = kalmar_state[kalmar_id].get("tracking_id")

                log_step(kalmar_id, "OCR_ATTEMPT", {
                    "tracking_id": tracking_id,
                    "attempt": attempt_count,
                    "since_pickup_sec": round(
                        time.time() - timer["start_time"], 2
                    )
                })

                # ----------------------------
                # CALL OCR API
                # ----------------------------
                response = requests.post(
                    OCR_API_URL,
                    json={
                        "kalmar_id": kalmar_id,
                        "tracking_id": tracking_id,
                        "action": "picked",
                        "timestamp": datetime.now().isoformat(),
                        "images": [image_base64]
                    },
                    timeout=20.0
                )

                log_step(kalmar_id, "OCR_API_CALL", {
                    "status": response.status_code,
                    "tracking_id": tracking_id
                })
                # ✅ ADD THIS BLOCK
                timer = self.pickup_timers.get(kalmar_id)
                if not timer or timer.get("stop"):
                    log_step(kalmar_id, "OCR_ABORT_AFTER_API_RETURN")
                    break


                # ----------------------------
                # PROCESS RESPONSE
                # ----------------------------
                if response.status_code == 200:
                    result = response.json()

                    if result.get("status") == "container_found":
                        container_number = result.get("container_number")

                        with self.container_lock:
                            if kalmar_id not in self.container_numbers:
                                self.container_numbers[kalmar_id] = {}

                            self.container_numbers[kalmar_id][container_number] = \
                                self.container_numbers[kalmar_id].get(container_number, 0) + 1

                            total_unique = len(self.container_numbers[kalmar_id])
                            total_count = self.container_numbers[kalmar_id][container_number]

                        log_step(kalmar_id, "CONTAINER_COLLECTED", {
                            "container": container_number,
                            "count": total_count,
                            "unique_total": total_unique
                        })
                        
                        # ✅ SEND SUCCESS TO BACKEND
                        if tracking_id:
                            failed_ocr_repo.save_success_attempt(
                                tracking_id=tracking_id,
                                image_bytes=image_bytes,
                                container_number=container_number,
                                kalmar_id=kalmar_id,
                                ocr_type="success"
                            )


                        log_step(kalmar_id, "CONTAINER_FOUND", {
                            "tracking_id": tracking_id,
                            "container_no": container_number
                        })

                        print(f"[{kalmar_id}] ✅ Container found: {container_number}")
                        # break

                    else:
                        # ----------------------------
                        # SAVE FAILED OCR ATTEMPT
                        # ----------------------------
                        if tracking_id:
                            failed_ocr_repo.save_failed_attempt(
                                tracking_id=tracking_id,
                                attempt_no=attempt_count,
                                image_bytes=image_bytes,
                                container_number=result.get("raw_text", "UNKNOWN"),
                                kalmar_id=kalmar_id
                            )

                else:
                    # API error also treated as fail
                    if tracking_id:
                        failed_ocr_repo.save_failed_attempt(
                            tracking_id=tracking_id,
                            attempt_no=attempt_count,
                            image_bytes=image_bytes,
                            container_number="API_ERROR",
                            kalmar_id=kalmar_id
                        )

            except Exception as e:
                print(f"[{kalmar_id}] ❌ OCR thread error: {e}")
                traceback.print_exc()
                time.sleep(1)



    # ============================================================
    # PLACED WITH DELAY (SINGLE LIVE FRAME)
    # ============================================================

    def handle_placed_with_delay(self, kalmar_id: str, placed_frame_ts: float):

    # ⛔ STOP PICKUP OCR IMMEDIATELY
        timer = self.pickup_timers.get(kalmar_id)
        if timer:
            timer["stop"] = True
        time.sleep(DELAY_SEC)

        tracking_id = kalmar_state[kalmar_id].get("tracking_id")
        # existing_container = self.get_container_number(kalmar_id)
        container_list = self.get_container_numbers(kalmar_id)
        # ====================================================
        # CASE 1: CONTAINER ALREADY FOUND DURING PICKUP
        # ====================================================
        if container_list:

            location = self.location_handler.get_location_at_time(
                kalmar_id,
                placed_frame_ts + int(DELAY_SEC * 1000)
            )

            if not location:
                log_step(kalmar_id, "PLACED_LOCATION_NOT_FOUND")
                return False

            # Send placed event with tracking_id
            self.placed_event_handler.send_placed_event(
                kalmar_id,
                container_list,
                location,
                tracking_id
            )

            log_step(kalmar_id, "PLACED_EVENT_SENT", {
                "tracking_id": tracking_id,
                "container_no": container_list
            })

            kalmar_state[kalmar_id]["placed_finalized"] = True
            self._reset_kalmar(kalmar_id)
            return True


        # ====================================================
        # CASE 2: NO CONTAINER → DO OCR ON PLACED FRAME
        # ====================================================

        live_frame = current_frames.get(kalmar_id)
        if live_frame is None:
            log_step(kalmar_id, "PLACED_NO_LIVE_FRAME")
            return False

        success, buffer = cv2.imencode(
            ".jpg",
            live_frame,
            [cv2.IMWRITE_JPEG_QUALITY, 100]
        )

        if not success:
            return False

        image_bytes = buffer.tobytes()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        container_no = None

        try:
            response = requests.post(
                OCR_API_URL,
                json={
                    "kalmar_id": kalmar_id,
                    "tracking_id": tracking_id,
                    "action": "placed",
                    "timestamp": datetime.now().isoformat(),
                    "images": [image_base64]
                },
                timeout=20.0
            )

            log_step(kalmar_id, "OCR_API_CALL_PLACED", {
                "status": response.status_code,
                "tracking_id": tracking_id
            })

            if response.status_code == 200:
                result = response.json()

                if result.get("status") == "container_found":
                    container_no = result.get("container_number")
                    
                    if tracking_id:
                        failed_ocr_repo.save_success_attempt(
                            tracking_id=tracking_id,
                            image_bytes=image_bytes,
                            container_number=container_no,
                            kalmar_id=kalmar_id,
                            ocr_type="success"
                        )

                else:
                    # SAVE FAILED PLACED OCR
                    if tracking_id:
                        failed_ocr_repo.save_failed_attempt(
                            tracking_id=tracking_id,
                            attempt_no=999,  # placed attempt marker
                            image_bytes=image_bytes,
                            container_number=result.get("raw_text", "UNKNOWN"),
                            kalmar_id=kalmar_id
                        )

            else:
                # API error also logged as fail
                if tracking_id:
                    failed_ocr_repo.save_failed_attempt(
                        tracking_id=tracking_id,
                        attempt_no=999,
                        image_bytes=image_bytes,
                        container_number="API_ERROR",
                        kalmar_id=kalmar_id
                    )

        except Exception as e:
            print(f"[{kalmar_id}] ❌ OCR error (placed): {e}")

            if tracking_id:
                failed_ocr_repo.save_failed_attempt(
                    tracking_id=tracking_id,
                    attempt_no=999,
                    image_bytes=image_bytes,
                    container_number="EXCEPTION",
                    kalmar_id=kalmar_id
                )


        # ====================================================
        # FALLBACK CONTAINER IF STILL NOT FOUND
        # ====================================================

        if not container_no:
            container_no = "UNKNOWN_CONTAINER"

        location = self.location_handler.get_location_at_time(
            kalmar_id,
            placed_frame_ts + int(DELAY_SEC * 1000)
        )

        if not location:
            log_step(kalmar_id, "PLACED_LOCATION_NOT_FOUND_AFTER_OCR")
            return False

        # Send placed event
        self.placed_event_handler.send_placed_event(
            kalmar_id,
            [container_no],
            location, tracking_id
        )

        log_step(kalmar_id, "PLACED_EVENT_SENT", {
            "tracking_id": tracking_id,
            "container_no": container_no
        })

        kalmar_state[kalmar_id]["placed_finalized"] = True
        self._reset_kalmar(kalmar_id)
        return True

        

        # ============================================================
        # RESET (UNCHANGED LOGIC)
        # ============================================================

    def _reset_kalmar(self, kalmar_id: str):
            log_step(kalmar_id, "RESET_STATE")

            kalmar_state[kalmar_id]["placed_delay_started"] = False
            kalmar_state[kalmar_id]["placed_finalized"] = False
            kalmar_state[kalmar_id]["tracking_id"] = None

            self.pickup_timers.pop(kalmar_id, None)
            with self.container_lock:
                self.container_numbers.pop(kalmar_id, None)
            self.ocr_results.pop(kalmar_id, None)
            self.picked_snapshots.pop(kalmar_id, None)


    # def get_container_number(self, kalmar_id: str):
    #         return self.container_numbers.get(kalmar_id)
    
    def get_container_numbers(self, kalmar_id: str):
        with self.container_lock:
            container_dict = self.container_numbers.get(kalmar_id, {})

            # Sort by detection count (highest first)
            sorted_containers = sorted(
                container_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )

            return [c[0] for c in sorted_containers]

    def get_best_container_number(self, kalmar_id: str, min_count: int = 2):
        with self.container_lock:
            container_dict = self.container_numbers.get(kalmar_id, {})

            if not container_dict:
                return None

            best_container, best_count = max(
                container_dict.items(),
                key=lambda x: x[1]
            )

            if best_count < min_count:
                return None

            return best_container

    def is_ocr_in_progress(self, kalmar_id: str):
            return (
                kalmar_id in self.pickup_timers
                and self.pickup_timers[kalmar_id]["triggered"]
            )

    def update_location(self, kalmar_id: str, location_data: dict):
            self.location_handler.update_location(kalmar_id, location_data)

    def get_location_at_android_time(self, kalmar_id: str, target_ts: float):
            return self.location_handler.get_location_at_time(kalmar_id, target_ts)




ocr_handler = OCRHandler()


# ================= MODEL =================
class CNN_LSTM_Industry(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # backbone = models.efficientnet_b0(
        #     weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        # )
        backbone = models.efficientnet_b0(weights=None)


        self.cnn = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.cnn_fc = nn.Sequential(
            nn.Linear(1280, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # -------- SAFE LSTM (NO CRASH) --------
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0.0
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        f = self.cnn(x)
        f = self.pool(f).view(f.size(0), -1)
        f = self.cnn_fc(f)

        f = f.view(B, T, -1)
        out, _ = self.lstm(f)

        out = out.mean(dim=1)
        return self.classifier(out)



def load_model():
    # ckpt = torch.load("kalmarModel_06_01_26.pth", map_location=DEVICE)
    ckpt = torch.load(
    "ForklifterV2.pth",
    map_location=DEVICE,
    weights_only=True
    )

    model = CNN_LSTM_Industry(len(CLASSES)).to(DEVICE)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    print(f"✅ Movement model loaded on {DEVICE}")
    return model



model = load_model()


class MKVRecorder:
    def __init__(self, kalmar_id: str):
        self.kalmar_id = kalmar_id
        self.process = None
        self.running = False

    def _build_output_path(self):
        date_str = datetime.now().strftime("%Y-%m-%d")
        base_dir = os.path.join(RECORD_BASE_DIR, self.kalmar_id, date_str)
        os.makedirs(base_dir, exist_ok=True)

        filename = f"{self.kalmar_id}_{datetime.now().strftime('%H%M%S')}_%03d.mkv"
        return os.path.join(base_dir, filename)

    def start(self):
        output_pattern = self._build_output_path()

        cmd = [
            "ffmpeg",
            "-loglevel", "error",

            "-f", "h264",
            "-threads", "2",
            "-i", "pipe:0",

            "-c", "copy",
            "-reset_timestamps", "1",

            "-f", "segment",
            "-segment_format", "matroska",
            "-segment_time", "600",
            "-fs", str(MAX_FILE_SIZE_BYTES),

            output_pattern
        ]

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**7
        )

        threading.Thread(
            target=self._read_stderr,
            daemon=True
        ).start()

        self.running = True
        print(f"[{self.kalmar_id}] 🎬 MKV recording started")

    def _read_stderr(self):
        for line in self.process.stderr:
            msg = line.decode(errors="ignore").strip()
            if msg:
                print(f"[{self.kalmar_id}][MKV] {msg}")

    def write(self, h264_bytes: bytes):
        if not self.running or not self.process:
            return
        try:
            self.process.stdin.write(h264_bytes)
            self.process.stdin.flush()
        except BrokenPipeError:
            self.stop()

    def stop(self):
        self.running = False
        if self.process:
            try:
                self.process.stdin.close()
                self.process.terminate()
            except Exception:
                pass
            self.process = None

        print(f"[{self.kalmar_id}] ⏹️ MKV recording stopped")



# ================= KALMAR SESSION =================
class KalmarSession:
    def __init__(self, kalmar_id):
        self.kalmar_id = kalmar_id
        self.h264_queue = queue.Queue(maxsize=20)
        self.frame_queue = queue.Queue(maxsize=20)
        self.running = True
        self.last_frame_time = time.time()
        self.ffmpeg = None
        self.frame_counter = 0
        self.h264_packet_counter = 0
        self.last_h264_time = time.time()
        self.recorder = MKVRecorder(kalmar_id)
        self.ffmpeg_failures = 0
        self.last_ffmpeg_start = 0
        self.ffmpeg_restarting = False
        self.encoder = None
        self.encoder_process = None
        self.broadcast_task = None
        self.encode_queue = queue.Queue(maxsize=5)
        self.init_segment = None  # Add this to store init segment
        self.capturing_init = True  # Flag to capture init segment
        self._pending_moof = None



        # self.recorder.start()


        if WEBSOCKET_ENABLED:
            workers.add(kalmar_id)
            print(f"[{kalmar_id}] Worker registered")

        # Start processing thread only (NO DISPLAY THREAD HERE)
        threading.Thread(target=self._processing_loop, daemon=True).start()
    
    def is_simple_h264_camera(self,rtsp_url: str) -> bool:
        return "ch01.264" in rtsp_url
   
    def start_ffmpeg(self, rtsp_url: str = None):
        print(f"[{self.kalmar_id}] Starting FFmpeg...")

        # 🔧 TESTING MODE: Use video file if enabled
        if USE_VIDEO_FILE:
            print(f"[{self.kalmar_id}] 🎥 VIDEO FILE MODE")
            if not os.path.exists(VIDEO_FILE_PATH):
                print(f"[{self.kalmar_id}] ❌ Video file not found: {VIDEO_FILE_PATH}")
                return
            
            cmd = [
                "ffmpeg",
                "-loglevel", "warning",
                
                # Read from video file
                #"-stream_loop", "-1",  # Loop video infinitely for testing
                "-i", VIDEO_FILE_PATH,
                
                "-vf", f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}:flags=lanczos",
                "-pix_fmt", "bgr24",
                "-f", "rawvideo",
                "pipe:1",
            ]
            
            print(f"[{self.kalmar_id}] 📹 Reading from: {VIDEO_FILE_PATH}")
            print(f"[{self.kalmar_id}] 🔁 Video will loop infinitely for testing")
            
        else:
            # Original RTSP camera mode
            simple_cam = self.is_simple_h264_camera(rtsp_url)

            try:
                if rtsp_url:

                    if simple_cam:
                        # ✅ SAFE MODE (ch01.264 cameras)
                        cmd = [
        "ffmpeg",
        "-loglevel", "warning",

        # ✅ HEVC stability flags
        "-fflags", "+nobuffer+discardcorrupt",
        "-err_detect", "ignore_err",
        "-flags", "low_delay",

        "-rtsp_transport", "tcp",
        "-rtsp_flags", "prefer_tcp",

        "-i", rtsp_url,

        # ✅ Drop broken frames
        "-vsync", "drop",

        "-vf", f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}:flags=lanczos",
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "pipe:1",
    ]

                    else:
                        # ✅ LOW-LATENCY MODE (real RTSP cameras)
                        cmd = [
        "ffmpeg",
        "-loglevel", "warning",

        # ✅ Explicit (fix warning also)
        "-hwaccel", "qsv",
        "-hwaccel_output_format", "qsv",
        "-c:v", "h264_qsv",

        "-rtsp_transport", "tcp",
        "-fflags", "+genpts",
        "-flags", "low_delay",

        "-analyzeduration", "1000000",
        "-probesize", "1000000",

        "-i", rtsp_url,

        # ✅ CORRECT FORMAT FLOW
        "-vf", f"scale_qsv=w={FRAME_WIDTH}:h={FRAME_HEIGHT},hwdownload,format=nv12,format=bgr24",

        "-f", "rawvideo",
        "pipe:1",
    ]


                else:
                    # (UNCHANGED) raw H264 pipe mode
                    cmd = [
                        "ffmpeg",
                        "-loglevel", "warning",
                        "-fflags", "+nobuffer+genpts",
                        "-flags", "low_delay",
                        "-avioflags", "direct",
                        "-f", "h264",
                        "-threads", "2",
                        "-i", "pipe:0",
                        "-vf", f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}:flags=lanczos",
                        "-pix_fmt", "bgr24",
                        "-f", "rawvideo",
                        "-flush_packets", "1",
                        "pipe:1",
                    ]

            except Exception as e:
                print(f"[{self.kalmar_id}] ERROR setting up FFmpeg command: {e}")
                return

        print(f"[{self.kalmar_id}] FFmpeg command: {' '.join(cmd)}")

        try:
            self.ffmpeg = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE if not rtsp_url and not USE_VIDEO_FILE else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8,
            )

            threading.Thread(target=self._read_ffmpeg_stderr, daemon=True).start()
            threading.Thread(target=self._frame_reader, daemon=True).start()
            self._start_h264_encoder()

        except Exception as e:
            print(f"[{self.kalmar_id}] ERROR starting FFmpeg: {e}")
            traceback.print_exc()
            self.stop()

    def _read_ffmpeg_stderr(self):
        if not self.ffmpeg or not self.ffmpeg.stderr:
            return

        try:
            while self.running and self.ffmpeg and self.ffmpeg.stderr:
                line = self.ffmpeg.stderr.readline()
                if line:
                    line_str = line.decode("utf-8", errors="ignore").strip()
                    if line_str and not line_str.startswith("frame="):
                        print(f"[{self.kalmar_id}][FFmpeg] {line_str}")
                else:
                    if self.ffmpeg and self.ffmpeg.poll() is not None:
                        print(
                            f"[{self.kalmar_id}] FFmpeg process ended with code: {self.ffmpeg.returncode}"
                        )
                        break
                    time.sleep(0.1)
        except Exception as e:
            if self.running:
                print(f"[{self.kalmar_id}] Error reading FFmpeg stderr: {e}")

    def _h264_writer(self):
        print(f"[{self.kalmar_id}] H264 writer started")

        while self.running:
            try:
                data = self.h264_queue.get(timeout=0.1)
                self.h264_packet_counter += 1
                self.last_h264_time = time.time()

                if self.h264_packet_counter <= 3:
                    print(
                        f"[{self.kalmar_id}] H264 packet {self.h264_packet_counter}, size: {len(data)} bytes"
                    )
                    if len(data) > 4:
                        if (
                            data[:4] == b"\x00\x00\x00\x01"
                            or data[:3] == b"\x00\x00\x01"
                        ):
                            print(f"[{self.kalmar_id}] H264 start code detected")

                if self.ffmpeg and self.ffmpeg.stdin:
                    try:
                        self.ffmpeg.stdin.write(data)
                        self.ffmpeg.stdin.flush()
                    except BrokenPipeError:
                        print(f"[{self.kalmar_id}] Broken pipe to FFmpeg")
                        break
                # 2️⃣ Send SAME packet to MKV recorder (NEW)
                if self.recorder:
                    self.recorder.write(data)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[{self.kalmar_id}] Error in H264 writer: {e}")
                break

        print(f"[{self.kalmar_id}] H264 writer stopped")

   

    def _frame_reader(self):
        frame_size = FRAME_WIDTH * FRAME_HEIGHT * 3
        print(
            f"[{self.kalmar_id}] Frame reader started, expecting {frame_size} bytes per frame"
        )

        consecutive_errors = 0

        #while self.running:
        while self.running and not shutdown_event.is_set():
            
            # ⏰ Watchdog: restart if no valid frame for 60 seconds
            if self.frame_counter > 0 and (time.time() - self.last_frame_time > 60):
                print(f"[{self.kalmar_id}] ⚠️ Stream frozen for 60 seconds")
                if not USE_VIDEO_FILE:  # Don't restart for video file
                    self._restart_ffmpeg(delay=5)
                return
            try:
                if not self.ffmpeg or not self.ffmpeg.stdout:
                    time.sleep(0.1)
                    continue

                frame_data = b""
                while len(frame_data) < frame_size:
                    chunk = self.ffmpeg.stdout.read(frame_size - len(frame_data))
                    if not chunk:
                        break
                    frame_data += chunk

                if len(frame_data) != frame_size:
                    consecutive_errors += 1
                    print(f"[{self.kalmar_id}] ⚠️ Incomplete frame ({len(frame_data)}) | errors={consecutive_errors}")

                    if consecutive_errors >= 30:   # 30 bad frames
                        print(f"[{self.kalmar_id}] ❌ Too many incomplete frames")
                        if not USE_VIDEO_FILE:  # Don't restart for video file
                            self._restart_ffmpeg(delay=5)
                        return

                    continue
                
                

              
                '''if not frame_data:
                    if self.ffmpeg and self.ffmpeg.poll() is not None:
                        print(f"[{self.kalmar_id}] ⚠️ FFmpeg died → reconnecting")
                        if not USE_VIDEO_FILE:  # Don't restart for video file
                            self._restart_ffmpeg()
                        return
                    time.sleep(0.01)
                    continue   '''
                
                if not frame_data:

                    # VIDEO FILE FINISHED
                    if USE_VIDEO_FILE:
                    
                        print(f"[{self.kalmar_id}] ✅ Video completed")

                        shutdown_application()

                        return
               

                consecutive_errors = 0

                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(
                    FRAME_HEIGHT, FRAME_WIDTH, 3
                )

                # ✅ Store for OCR (MANDATORY)
                current_frames[self.kalmar_id] = frame

                # ✅ RTSP-only timestamp (SINGLE SOURCE OF TRUTH)
                frame_ts = monotonic_ms()

                # ✅ Store for OCR & processing
                current_frames[self.kalmar_id] = frame

                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()  # drop OLD frame
                    except:
                        pass

                self.frame_queue.put_nowait((frame, frame_ts))

                
                
                

                self.last_frame_time = time.time()
                self.frame_counter += 1

                if self.frame_counter == 1:
                    print(f"[{self.kalmar_id}] ✅ FIRST FRAME RECEIVED!")
                    print(f"[{self.kalmar_id}] Frame shape: {frame.shape}")
                    cv2.imwrite(f"first_frame_{self.kalmar_id}.jpg", frame)
                    print(
                        f"[{self.kalmar_id}] Saved first frame to first_frame_{self.kalmar_id}.jpg"
                    )

                if self.frame_counter % 100 == 0:
                    print(f"[{self.kalmar_id}] Received {self.frame_counter} frames")

            except Exception as e:
                print(f"[{self.kalmar_id}] Error in frame reader: {e}")
                traceback.print_exc()
                time.sleep(0.1)

        print(f"[{self.kalmar_id}] Frame reader stopped")
        
    def _restart_ffmpeg(self, delay=5):
        if not self.running:
            return

        if self.ffmpeg_restarting:
            return

        self.ffmpeg_restarting = True
        self.ffmpeg_failures += 1

        print(f"[{self.kalmar_id}] 🔄 Restarting FFmpeg in {delay} seconds...")

        try:
            if self.ffmpeg:
                try:
                    self.ffmpeg.kill()
                except Exception:
                    pass
                self.ffmpeg = None
                
                        # ✅ IMPORTANT RESET
            self.frame_counter = 0
            self.last_frame_time = time.time()

            time.sleep(delay)   # ⏰ 2 minute retry delay

            self.start_ffmpeg(rtsp_url=SINGLE_RTSP_URL if not USE_VIDEO_FILE else None)

        finally:
            self.ffmpeg_restarting = False
            
            
    def _start_h264_encoder(self):

        if self.encoder_process:
            return

        print(f"[{self.kalmar_id}] Starting fMP4 encoder...")

        cmd = [
            "ffmpeg",
            "-loglevel", "error",

            # ✅ RESET TIMESTAMPS
            "-fflags", "+genpts",
            "-use_wallclock_as_timestamps", "1",

            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{DISPLAY_WIDTH}x{DISPLAY_HEIGHT}",
            "-r", "20",
            "-i", "pipe:0",

            "-an",
           # "-c:v", "libx264",
            "-c:v", "h264_qsv",
            "-preset", "veryfast",
            "-look_ahead", "0",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-profile:v", "baseline",
            "-level", "3.0",

            "-pix_fmt", "yuv420p",

            "-g", "60",
            "-keyint_min", "60",
            "-sc_threshold", "0",

            # ✅ IMPORTANT FOR MSE
            "-movflags", "frag_keyframe+empty_moov+default_base_moof+faststart",
            "-frag_duration", "500000",
            "-f", "mp4",
            "pipe:1",
        ]

        self.encoder_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        threading.Thread(
            target=self._encoder_writer,
            daemon=True
        ).start()

        threading.Thread(
            target=self._broadcast_loop,
            daemon=True
        ).start()








    def _broadcast_loop(self):
        print(f"[{self.kalmar_id}] Broadcast loop started")

        buffer = bytearray()

        #while self.running and self.encoder_process:
        while self.running and self.encoder_process and not shutdown_event.is_set():
            try:
                chunk = self.encoder_process.stdout.read(4096)

                if not chunk:
                    time.sleep(0.001)
                    continue

                buffer.extend(chunk)

                # ✅ Parse complete MP4 boxes
                while len(buffer) >= 8:
                    box_size = int.from_bytes(buffer[0:4], byteorder="big")

                    # Incomplete box
                    if box_size <= 0 or len(buffer) < box_size:
                        break

                    box_type = buffer[4:8].decode(errors="ignore")
                    box_data = buffer[:box_size]

                    # print(f"[{self.kalmar_id}] 📦 Sending box: {box_type} ({box_size} bytes)")

                    # Remove box from buffer
                    buffer = buffer[box_size:]

                    # ✅ Capture init segment (ftyp + moov)
                    if self.capturing_init:
                        if box_type in ("ftyp", "moov"):
                            if not self.init_segment:
                                self.init_segment = bytearray()

                            self.init_segment.extend(box_data)

                            if box_type == "moov":
                                self.capturing_init = False
                                print(
                                    f"[{self.kalmar_id}] ✅ Captured init segment: "
                                    f"{len(self.init_segment)} bytes"
                                )
                        continue

                    # ✅ Send full box to clients
                    # ✅ After init is captured → combine moof + mdat

                if box_type == "moof":
                    # Store moof temporarily
                    self._pending_moof = box_data

                elif box_type == "mdat" and self._pending_moof:
                    # Combine moof + mdat into one fragment
                    fragment = self._pending_moof + box_data
                    self._pending_moof = None

                    # print(
                    #     f"[{self.kalmar_id}] 📤 Sending fragment: "
                    #     f"{len(fragment)} bytes"
                    # )

                    if ASYNC_LOOP and ASYNC_LOOP.is_running():
                        asyncio.run_coroutine_threadsafe(
                            self._send_to_clients(fragment),
                            ASYNC_LOOP
        )


            except Exception as e:
                print(f"[{self.kalmar_id}] Broadcast error: {e}")
                time.sleep(0.01)

        
    async def _send_to_clients(self, chunk: bytes):
            async with clients_lock:
                client_set = clients.get(self.kalmar_id, set())
                
                dead = []
                for ws in client_set:
                    try:
                        # Check if this is a new client that needs init segment
                        client_id = id(ws)
                        if not hasattr(self, '_clients_initted'):
                            self._clients_initted = {}
                        
                        # Send init segment first for new clients
                        if client_id not in self._clients_initted and self.init_segment:
                            # print(f"[{self.kalmar_id}] Sending init segment to new client: {len(self.init_segment)} bytes")
                            await ws.send_bytes(bytes(self.init_segment))
                            self._clients_initted[client_id] = True
                        # print(f"[{self.kalmar_id}] 📤 Sending chunk ------------: {len(chunk)} bytes")
                        # Send the current chunk
                        await ws.send_bytes(chunk)
                    except Exception as e:
                        print(f"Error sending to client: {e}")
                        dead.append(ws)
                
                for ws in dead:
                    client_set.discard(ws)

    
    
    
    def _encoder_writer(self):

        print(f"[{self.kalmar_id}] Encoder writer started")

        expected_size = DISPLAY_WIDTH * DISPLAY_HEIGHT * 3

        #while self.running and self.encoder_process:
        while self.running and self.encoder_process and not shutdown_event.is_set():

            try:
                frame = self.encode_queue.get(timeout=0.1)

                if frame is None:
                    continue

                # ✅ HARD VALIDATION
                if frame.shape[0] != DISPLAY_HEIGHT or \
                frame.shape[1] != DISPLAY_WIDTH or \
                frame.shape[2] != 3:
                    print(f"[{self.kalmar_id}] ❌ Invalid frame shape {frame.shape}")
                    continue

                raw = frame.tobytes()

                if len(raw) != expected_size:
                    print(f"[{self.kalmar_id}] ❌ Invalid frame size {len(raw)}")
                    continue

                if self.encoder_process.stdin:
                    self.encoder_process.stdin.write(raw)
                    self.encoder_process.stdin.flush()

            except queue.Empty:
                continue

            except BrokenPipeError:
                print(f"[{self.kalmar_id}] Encoder broken pipe → restarting")
                self._restart_encoder()
                break

            except Exception as e:
                print(f"[{self.kalmar_id}] Encoder writer error: {e}")
                break

        print(f"[{self.kalmar_id}] Encoder writer stopped")

   
   
   
    def _restart_encoder(self):
        try:
            if self.encoder_process:
                self.encoder_process.kill()
        except:
            pass

        self.encoder_process = None
        time.sleep(1)
        self._start_h264_encoder()



    last_state_debug_time = 0

    def _processing_loop(self):
        print(f"[{self.kalmar_id}] Processing loop started")

        ws_frame_counter = 0
        last_fps_time = time.time()
        frames_since_last = 0
        last_placed_check_time = 0
        last_state_debug_time = 0  # ✅ ADD THIS
        last_process_time = time.time()
        target_fps = 15.0
        frame_interval = 1.0 / target_fps

        frame_quality_check_counter = 0

        #while self.running:
        while self.running and not shutdown_event.is_set():
            try:
                current_time = time.time()
                elapsed = current_time - last_process_time
                sleep_needed = max(0, frame_interval - elapsed)
                if sleep_needed > 0:
                    time.sleep(sleep_needed)
                last_process_time = time.time()

                # -------- FRAME FETCH (unchanged) --------
                frame = None
                frame_ts = None
                while not self.frame_queue.empty():
                    frame, frame_ts = self.frame_queue.get_nowait()
                if frame is None:
                    continue
                
                state = kalmar_state[self.kalmar_id]
                
                # now = time.time()
                # if now - last_state_debug_time >= 2.0:
                #     last_state_debug_time = now

                #     debug_state = {
                #         "fsm_state": state["fsm"].state,
                #         "visible_state": (
                #             "placed" if state.get("temp_placed_active") else state["fsm"].state
                #         ),
                #         "confidence": round(state.get("confidence", 0.0), 3),
                #     }

                #     log_step(self.kalmar_id, "STATE_DEBUG", debug_state)
                #     print(f"[{self.kalmar_id}] debug_state: {debug_state}")

                
                state["current_frame_ts"] = frame_ts


                frames_since_last += 1
                current_time = time.time()

                # ocr_handler.store_frame_timestamp(self.kalmar_id, current_time)

                if current_time - last_fps_time >= 0.5:
                    fps = frames_since_last / (current_time - last_fps_time)
                    kalmar_state[self.kalmar_id]["fps_counter"].append(fps)
                    frames_since_last = 0
                    last_fps_time = current_time

                state = kalmar_state[self.kalmar_id]

                # ✅ CHANGE 1: ALWAYS read action from FSM (not prev_action / visible)
                current_action = state["fsm"].state
                current_confidence = state["confidence"]

                # -------- PICKUP TRACKING (unchanged logic) --------
                pickup_triggered, pickup_duration = ocr_handler.track_pickup(
                    self.kalmar_id,
                    current_action,
                    current_confidence,
                    frame,
                )

                # -------- PLACED HANDLING (FIXED) --------
                if current_time - last_placed_check_time >= 0.1:

                    # ✅ CHANGE 2: placed comes ONLY from FSM temp flag
                    # if state["temp_placed_active"]:

                    # if state["temp_placed_active"] and not state.get("placed_delay_started", False):
                    # ⛔ HARD GUARD: placed already done for this cycle
                    if state.get("placed_finalized"):
                        continue

                    if state["temp_placed_active"] and not state.get("placed_delay_started", False):


                        placed_frame_ts = state.get("current_frame_ts")                       

                        log_step("PLACED_DETECTED", {
                            "kalmar_id": self.kalmar_id,
                            "frame_ts_android_ms": placed_frame_ts,
                            "frame_time_human": datetime.now().isoformat()
                        })



                        if placed_frame_ts is None:
                            log_step(self.kalmar_id, "PLACED_NO_FRAME_TS")
                            # last_placed_check_time = current_time
                            continue
                        # if placed_frame_ts is None:
                        #     log_step(self.kalmar_id, "PLACED_TS_FALLBACK", {
                        #         "fallback_ts": placed_frame_ts,
                        #         "source": "system_time"
                        #     })
                        #     # placed_frame_ts = time.time() * 1000  # ms
                        #     last_placed_check_time = current_time
                        #     continue

                        state["placed_delay_started"] = True

                        state["placed_frame_ts"] = placed_frame_ts

                        log_step(self.kalmar_id, "PLACED_FRAME_CAPTURED", {
                            "frame_ts": placed_frame_ts
                        })

                        # threading.Thread(
                        #     target=ocr_handler.handle_placed_with_delay,
                        #     args=(self.kalmar_id, placed_frame_ts),
                        #     daemon=True
                        # ).start()
                        
                        OCR_EXECUTOR.submit(
                            ocr_handler.handle_placed_with_delay,
                            self.kalmar_id,
                            placed_frame_ts
                        )

                        
                        
                        state["temp_placed_active"] = False


                        # threading.Thread(
                        #     target=ocr_handler.handle_placed_with_delay,
                        #     args=(self.kalmar_id, current_time),
                        #     daemon=True
                        # ).start()

                    
                    # if state["temp_placed_active"]:

                    #     pickup_timer = ocr_handler.pickup_timers.get(self.kalmar_id)
                    #     if pickup_timer and "first_ocr_time" in pickup_timer:
                    #         ocr_elapsed = current_time - pickup_timer["first_ocr_time"]

                    #         # ⏰ Wait for llama.cpp
                    #         if ocr_elapsed < MIN_OCR_WAIT_AFTER_FIRST:
                    #             # still waiting → do NOT place yet
                    #             pass
                    #         else:
                    #             ocr_handler.handle_placed_action(
                    #                 self.kalmar_id,
                    #                 "placed",
                    #                 current_confidence,
                    #                 current_time
                    #             )
                    #     else:
                    #         # OCR never started → allow placed normally
                    #         ocr_handler.handle_placed_action(
                    #             self.kalmar_id,
                    #             "placed",
                    #             current_confidence,
                    #             current_time
                    #         )


                    last_placed_check_time = current_time

                # -------- PICKUP DURATION RESET (unchanged intent) --------
                if state["fsm"].state != "picked":
                    pickup_duration = 0.0

                # -------- DISPLAY FRAME (unchanged) --------
                display_frame = cv2.resize(
                    frame,
                    (DISPLAY_WIDTH, DISPLAY_HEIGHT),
                    interpolation=cv2.INTER_LINEAR
                )

                # ✅ Start H264 encoder once
                # self._start_h264_encoder()


                frame_quality_check_counter += 1
                if frame_quality_check_counter % 150 == 0:
                    gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
                    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
                    if fm < 100:
                        kernel = np.array([
                            [-1, -1, -1],
                            [-1,  9, -1],
                            [-1, -1, -1]
                        ])
                        display_frame = cv2.filter2D(display_frame, -1, kernel)

                fps_display = (
                    sum(state["fps_counter"]) / len(state["fps_counter"])
                    if state["fps_counter"] else 0
                )

                visible_state = (
                    "placed"
                    if state["temp_placed_active"]
                    else state["fsm"].state
                )

                best_container = ocr_handler.get_best_container_number(self.kalmar_id)

                # Optional fallback: show first unique OCR read if a stable best value
                # is not yet available, without changing the OCR storage logic.
                if not best_container:
                    container_list = ocr_handler.get_container_numbers(self.kalmar_id)
                    if container_list:
                        best_container = list(dict.fromkeys(container_list))[0]

                display_frame = draw_status_overlay(
                    display_frame,
                    visible_state,
                    current_confidence,
                    kalmar_id=self.kalmar_id,
                    fps_display=fps_display,
                    container_text=best_container,
                    pickup_duration=pickup_duration if pickup_duration > 0 else None,
                    ocr_in_progress=ocr_handler.is_ocr_in_progress(self.kalmar_id),
                )


                # ✅ Send frame to H264 encoder
                if self.encoder_process:
                    try:
                        if self.encode_queue.full():
                            try:
                                self.encode_queue.get_nowait()  # drop old frame
                            except queue.Empty:
                                pass

                        self.encode_queue.put_nowait(display_frame.copy())

                    except queue.Full:
                        pass


                with display_lock:
                    global_display_frames[self.kalmar_id] = display_frame                

            except Exception as e:
                print(f"[{self.kalmar_id}] Processing error: {e}")
                traceback.print_exc()
                time.sleep(0.1)



        print(f"[{self.kalmar_id}] Processing loop stopped")

    def stop(self):
        self.running = False

        ocr_handler._reset_kalmar(self.kalmar_id)

        if self.kalmar_id in current_frames:
            del current_frames[self.kalmar_id]

        # Remove from global display frames
        with display_lock:
            if self.kalmar_id in global_display_frames:
                del global_display_frames[self.kalmar_id]

        if WEBSOCKET_ENABLED:
            if self.kalmar_id in workers:
                workers.remove(self.kalmar_id)
            # if self.kalmar_id in latest_frame_base64:
            #     del latest_frame_base64[self.kalmar_id]

        if self.encoder_process:
            try:
                self.encoder_process.kill()
            except:
                pass

        if self.ffmpeg:
            try:
                self.ffmpeg.kill()
            except BaseException:
                pass
            
        if self.recorder:
            self.recorder.stop()
            
        # for s in aux_camera_sessions.values():
        #     s.stop()
    
    

        if self.kalmar_id in kalmar_sessions:
            del kalmar_sessions[self.kalmar_id]

        print(f"[{self.kalmar_id}] Stopped")


# ================= WebSocket Functions =================
async def broadcast_to_clients(kalmar_id: str, frame_base64: str):
    # if not WEBSOCKET_ENABLED:
    #     return

    with clients_lock:
        client_set = clients.get(kalmar_id, set())
        if not client_set:
            return

        to_remove = []
        for ws in list(client_set):
            try:
                await ws.send_text(frame_base64)
            except BaseException:
                to_remove.append(ws)

        for ws in to_remove:
            client_set.discard(ws)


# ================= MODEL PROCESSING LOOP =================
def model_loop():
    model_interval = 1.0 / TARGET_MODEL_FPS

    # FSM persistence (seconds)
    MIN_PERSIST = {
        ("normal", "picked"): 0.25,
        ("picked", "normal"): 0.2,
    }

    #while True:
    while not shutdown_event.is_set():
        try:
            now = time.time()

            for kalmar_id, session in list(kalmar_sessions.items()):
                state = kalmar_state[kalmar_id]

                # ================= FPS THROTTLE =================
                if now - state["last_model_time"] < model_interval:
                    continue

                if not session or session.frame_queue.empty():
                    continue

                # frame = session.frame_queue.queue[-1]
                # frame, _ = session.frame_queue.queue[-1]
                try:
                    frame, _ = session.frame_queue.get_nowait()
                except queue.Empty:
                    continue



                state["last_model_time"] = now

                # ================= PREPROCESS =================
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = transform(rgb)  # (C, H, W)

                state["buffer"].append(tensor)
                if len(state["buffer"]) < SEQ_LEN:
                    continue

                tensor_seq = (
                    torch.stack(list(state["buffer"]))
                    .unsqueeze(0)
                    .to(DEVICE)
                )  # (1, T, C, H, W)

                # ================= MODEL INFERENCE =================
                with MODEL_LOCK, torch.no_grad():
                    logits = model(tensor_seq)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

                best_idx = int(np.argmax(probs))
                best_class = CLASSES[best_idx]
                best_conf = float(probs[best_idx])

                # ================= RAW DECISION (2-CLASS ONLY) =================
                if best_conf >= CLASS_THRESHOLDS[best_class]:
                    predicted = best_class
                else:
                    predicted = "normal"

                # ================= TEMPORAL SMOOTHING =================
                smoother = state["smoother"]
                smooth_pred, smooth_conf = smoother.update(predicted, probs)

                # ================= FSM =================
                fsm = state["fsm"]
                elapsed = time.time() - smoother.last_change_time

                if fsm.can_transition(smooth_pred):
                    required = MIN_PERSIST.get((fsm.state, smooth_pred), 0.5)
                    if elapsed >= required:
                        fsm.force(smooth_pred)

                # ================= TEMP PLACED LOGIC =================
                now_ts = time.time()

                # Track last picked
                if fsm.state == "picked":
                    state["last_picked_time"] = now_ts
                    state["normal_since"] = None
                    state["temp_placed_emitted"] = False

                # Detect picked → normal → placed
                if fsm.state == "normal" and state["last_picked_time"] is not None:
                    if state["normal_since"] is None:
                        state["normal_since"] = now_ts

                    elapsed_normal = now_ts - state["normal_since"]

                    if (
                        elapsed_normal >= WAIT_BEFORE_TEMP
                        and not state["temp_placed_emitted"]
                    ):
                        state["temp_placed_active"] = True
                        state["temp_placed_until"] = now_ts + TEMP_PLACED_DURATION
                        state["temp_placed_emitted"] = True

                # Expire placed
                if (
                    state["temp_placed_active"]
                    and now_ts >= state["temp_placed_until"]
                ):
                    state["temp_placed_active"] = False
                    state["placed_delay_started"] = False  # ✅ prevent re-fire

                # ================= FINAL VISIBLE STATE =================
                visible_state = (
                    "placed"
                    if state["temp_placed_active"]
                    else fsm.state
                )

                # state["prev_action"] = visible_state
                state["prev_action"] = fsm.state

                state["confidence"] = smooth_conf
                state["current_prediction"] = (
                    f"{visible_state} ({smooth_conf:.2f})"
                )

            time.sleep(0.01)

        except Exception as e:
            print(f"[Model Loop] Error: {e}")
            time.sleep(0.01)


            
            
            
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(
            LOG_DIR,
            f"kalmar_{datetime.now().strftime('%Y%m%d')}.log"
        )

_log_lock = threading.Lock()

# ================= LOGS TO ADD =================
def log_step(kalmar_id: str, step: str, data: dict = None):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    prefix = f"[{ts}][{kalmar_id}][{step}]"

    if data:
        msg = f"{prefix} {json.dumps(data, default=str)}"
    else:
        msg = prefix

    # ✅ Print to console
    print(msg)

    # ✅ Append to file (thread-safe)
    with _log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()


# ============================================================
# TRACKING MANAGER (PRODUCTION SAFE)
# ============================================================

class TrackingManager:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.counter_file = os.path.join(base_dir, "tracking_counter.json")
        self.lock = threading.Lock()

    def _load_counter(self):
        if not os.path.exists(self.counter_file):
            return {}

        try:
            with open(self.counter_file, "r") as f:
                return json.load(f)
        except:
            return {}

    def _save_counter(self, data):
        with open(self.counter_file, "w") as f:
            json.dump(data, f)

    def generate_tracking_id(self, kalmar_id: str):
        with self.lock:
            today_key = datetime.now().strftime("%Y-%m-%d")
            day_str = datetime.now().strftime("%a_%d_%m_%y").upper()

            data = self._load_counter()

            if today_key not in data:
                data = {today_key: 0}

            data[today_key] += 1
            counter = data[today_key]

            self._save_counter(data)

            safe_kalmar = kalmar_id.replace(" ", "").upper()
            return f"{safe_kalmar}_{day_str}_{counter:03d}"


# ============================================================
# FAILED OCR STORAGE (PRODUCTION SAFE)
# ============================================================

class FailedOCRRepository:
    def __init__(self, server_api: str, local_api: str):
        self.server_api = server_api
        self.local_api = local_api

    def save_failed_attempt(
        self,
        tracking_id: str,
        attempt_no: int,
        image_bytes: bytes,
        container_number: str,
        kalmar_id: str
    ):
        try:
            files = {
                "image": (
                    f"attempt_{attempt_no}.jpg",
                    image_bytes,
                    "image/jpeg"
                )
            }

            data = {
                "tracking_id": tracking_id,
                "attempt_no": attempt_no,
                "kalmar_id": kalmar_id,
                "container_number": container_number,
                "status": "FAIL",
                "timestamp": datetime.now().isoformat()
            }

            # -------------------------
            # 1️⃣ TRY MAIN SERVER
            # -------------------------
            try:
                response = requests.post(
                    self.server_api + "failed-ocr",
                    files=files,
                    data=data,
                    timeout=5
                )

                if response.status_code == 200:
                    log_step("FAILED_OCR_SERVER_SUCCESS", {
                        "tracking_id": tracking_id,
                        "attempt": attempt_no
                    })
                    return

                raise Exception(f"HTTP {response.status_code}")

            except Exception as e:
                log_step("FAILED_OCR_SERVER_FAILED", {
                    "tracking_id": tracking_id,
                    "error": str(e)
                })

            # -------------------------
            # 2️⃣ FALLBACK → LOCAL
            # -------------------------
            try:
                response = requests.post(
                    self.local_api + "failed-ocr",
                    files=files,
                    data=data,
                    timeout=5
                )

                if response.status_code == 200:
                    log_step("FAILED_OCR_LOCAL_QUEUED", {
                        "tracking_id": tracking_id,
                        "attempt": attempt_no
                    })
                    return

                log_step("FAILED_OCR_LOCAL_FAILED_STATUS", {
                    "status": response.status_code
                })

            except Exception as e:
                log_step("FAILED_OCR_LOCAL_ERROR", {
                    "error": str(e)
                })

        except Exception as e:
            print(f"[FAILED_OCR] Fatal error: {e}")

    def save_success_attempt(
        self,
        tracking_id: str,
        image_bytes: bytes,
        container_number: str,
        kalmar_id: str,
        ocr_type: str = "Success"
    ):
        try:
            files = {
                "image": (
                    f"{tracking_id}_{container_number}_success.jpg",
                    image_bytes,
                    "image/jpeg"
                )
            }

            data = {
                "tracking_id": tracking_id,
                "kalmar_id": kalmar_id,
                "container_number": container_number,
                "status": "S",
                "ocr_type": ocr_type,
                "timestamp": datetime.now().isoformat()
            }

            # ✅ 1️⃣ TRY MAIN SERVER
            try:
                response = requests.post(
                    self.server_api + "success-ocr",
                    files=files,
                    data=data,
                    timeout=5
                )

                if response.status_code == 200:
                    log_step(kalmar_id, "SUCCESS_OCR_SERVER_SUCCESS", {
                        "tracking_id": tracking_id,
                        "container": container_number
                    })
                    return

                raise Exception(f"HTTP {response.status_code}")

            except Exception as e:
                log_step(kalmar_id, "SUCCESS_OCR_SERVER_FAILED", {
                    "tracking_id": tracking_id,
                    "error": str(e)
                })

            # ✅ 2️⃣ FALLBACK → LOCAL
            try:
                response = requests.post(
                    self.local_api + "success-ocr",
                    files=files,
                    data=data,
                    timeout=5
                )

                if response.status_code == 200:
                    log_step(kalmar_id, "SUCCESS_OCR_LOCAL_QUEUED", {
                        "tracking_id": tracking_id
                    })
                    return

                log_step(kalmar_id, "SUCCESS_OCR_LOCAL_FAILED_STATUS", {
                    "status": response.status_code
                })

            except Exception as e:
                log_step(kalmar_id, "SUCCESS_OCR_LOCAL_ERROR", {
                    "error": str(e)
                })

        except Exception as e:
            print(f"[SUCCESS_OCR] Fatal error: {e}")








tracking_manager = TrackingManager(BASE_DIR)
failed_ocr_repo = FailedOCRRepository(SERVER_API, LOCAL_API)








# ================= MAIN DISPLAY LOOP =================
def main_display_loop():
    """THIS MUST RUN IN THE MAIN THREAD - OpenCV requirement"""
    windows_created = {}

    print("\n🖥️ Starting MAIN THREAD video display loop...")

    while True:
        try:
            # Get frames from global display frames
            frames_to_display = {}
            with display_lock:
                frames_to_display = global_display_frames.copy()

            # If no frames, show waiting message
            if not frames_to_display:
                # Create a simple main window
                window_name = "Kalmar Monitor"
                if window_name not in windows_created:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)
                    windows_created[window_name] = True

                # Create waiting frame
                waiting_frame = np.zeros(
                    (DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8
                )
                cv2.putText(
                    waiting_frame,
                    "KALMAR MONITOR",
                    (DISPLAY_WIDTH // 2 - 150, DISPLAY_HEIGHT // 2 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    waiting_frame,
                    "Waiting for streams...",
                    (DISPLAY_WIDTH // 2 - 140, DISPLAY_HEIGHT // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    waiting_frame,
                    "Press 'q' to quit",
                    (DISPLAY_WIDTH // 2 - 100, DISPLAY_HEIGHT // 2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )

                cv2.imshow(window_name, waiting_frame)

            # Create windows and display frames
            for kalmar_id, frame in frames_to_display.items():
                window_name = f"Kalmar {kalmar_id}"

                # Create window if not exists
                if kalmar_id not in windows_created:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)
                    windows_created[kalmar_id] = True
                    print(f"✅ Created window in MAIN THREAD: {window_name}")

                # Display frame
                cv2.imshow(window_name, frame)

            # Handle keyboard (VERY IMPORTANT: must be in main thread)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n🛑 Quitting all streams...")
                # Stop all sessions
                for kalmar_id in list(kalmar_sessions.keys()):
                    if kalmar_id in kalmar_sessions:
                        kalmar_sessions[kalmar_id].stop()
                break
            elif key == ord("s"):
                # Save screenshot
                if frames_to_display:
                    kalmar_id, frame = list(frames_to_display.items())[0]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_{kalmar_id}_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Screenshot saved: {filename}")
            elif key == ord("d"):
                # Debug info
                ocr_handler.location_handler.dump_locations(kalmar_id)

                print("\n=== DEBUG INFO ===")
                print(f"Active kalmars: {list(kalmar_sessions.keys())}")
                print(f"Display frames: {list(global_display_frames.keys())}")
                print(f"Current frames: {list(current_frames.keys())}")
                for kalmar_id, session in kalmar_sessions.items():
                    if session:
                        print(f"\n[{kalmar_id}]")
                        print(f"  H264 queue size: {session.h264_queue.qsize()}")
                        print(f"  Frame queue size: {session.frame_queue.qsize()}")
                        print(f"  Frame counter: {session.frame_counter}")
                        print(f"  H264 counter: {session.h264_packet_counter}")
                        # Show location history info
                        if hasattr(ocr_handler.location_handler, 'location_history'):
                            history = ocr_handler.location_handler.location_history.get(kalmar_id, [])
                            if history:
                                print(f"  Location history: {len(history)} entries")
                                latest = history[-1][1] if history else None
                                if latest:
                                    print(f"  Latest location timestamp: {latest['timestamp']}")
                print("=================\n")

            # Small delay
            time.sleep(0.001)

        except KeyboardInterrupt:
            print("\n⌨️ Keyboard interrupt received")
            break
        except Exception as e:
            print(f"[Display] Error: {e}")
            traceback.print_exc()
            time.sleep(0.1)


# ================= FASTAPI SETUP =================
app = FastAPI(title="Kalmar FFmpeg Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class KalmarConfig(BaseModel):
    kalmarId: str
    rtspUrl: str
    width: Optional[int] = FRAME_WIDTH
    height: Optional[int] = FRAME_HEIGHT



@app.websocket("/ws-ingestNew/{kalmar_id}")
async def ws_ingest(ws: WebSocket, kalmar_id: str):
    
    if not WEBSOCKET_ENABLED:
     await ws.close(code=1003, reason="WebSocket ingest disabled")
     return

    await ws.accept()

    # ---- Restart existing session if any ----
    if kalmar_id in kalmar_sessions:
        print(f"[INGEST] Restarting existing session for {kalmar_id}")
        kalmar_sessions[kalmar_id].stop()
        time.sleep(1)

    session = KalmarSession(kalmar_id)
    kalmar_sessions[kalmar_id] = session
    # session.start_ffmpeg()

    print(f"[INGEST] Client connected for {kalmar_id}")

    packet_count = 0
    total_bytes = 0
    start_time = time.time()
    last_packet_time = time.time()

    try:
        while True:
            msg = await ws.receive()

            # ✅ VERY IMPORTANT: handle disconnect FIRST
            if msg["type"] == "websocket.disconnect":
                print(f"[INGEST] Client disconnected for {kalmar_id}")
                break

            if msg["type"] != "websocket.receive":
                continue

            last_packet_time = time.time()

            # ================= BINARY (H264) =================
            if msg.get("bytes"):
                data = msg["bytes"]
                packet_count += 1
                total_bytes += len(data)

                try:
                    if not session.encoder_process:
                        session._start_h264_encoder()
                    session.h264_queue.put(data, timeout=0.01)
                except queue.Full:
                    # Drop oldest packet (low latency)
                    try:
                        session.h264_queue.get_nowait()
                    except queue.Empty:
                        pass
                    session.h264_queue.put_nowait(data)

                if packet_count % 100 == 0:
                    elapsed = time.time() - start_time
                    kbps = (total_bytes * 8 / 1000) / elapsed if elapsed > 0 else 0
                    print(
                        f"[INGEST][{kalmar_id}] Stats: {packet_count} packets, "
                        f"{total_bytes/1024:.1f} KB, {kbps:.1f} kbps"
                    )

            # ================= TEXT (METADATA) =================
            elif msg.get("text"):
                try:
                    data = json.loads(msg["text"])
                    msg_type = data.get("type")
                   
                    # ---- Location metadata ----
                    if msg_type == "location":
                        location_data = {
                            "latitude": float(data["latitude"]),
                            "longitude": float(data["longitude"]),
                            "timestamp": float(data["timestamp"]),
                        }

                        ocr_handler.update_location(kalmar_id, location_data)

                        # print(f"[location_data][{location_data}]")
                        log_step(kalmar_id, "LOCATION_RECEIVED", {
                            "lat": location_data["latitude"],
                            "lon": location_data["longitude"],
                            "ts": location_data["timestamp"],
                        })

                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    print(f"[INGEST][{kalmar_id}] JSON error: {e}")

            # ================= IDLE TIMEOUT (SAFETY) =================
            if time.time() - last_packet_time > 10:
                print(f"[INGEST] Idle timeout for {kalmar_id}")
                break

    except (WebSocketDisconnect, ConnectionResetError):
        print(f"[INGEST] Remote closed connection for {kalmar_id}")

    except Exception as e:
        print(f"[INGEST] Unexpected error for {kalmar_id}: {e}")
        traceback.print_exc()

    finally:
        if kalmar_id in kalmar_sessions:
            print(f"[INGEST] Stopping session for {kalmar_id}")
            kalmar_sessions[kalmar_id].stop()








@app.websocket("/ws/{kalmar_id}")
async def ws_kalmar(websocket: WebSocket, kalmar_id: str):

    await websocket.accept()

    async with clients_lock:
        clients.setdefault(kalmar_id, set()).add(websocket)

    print(f"[WS] Viewer connected to {kalmar_id}")

    # ✅ Send saved init segment immediately
    kalmar = kalmar_sessions.get(kalmar_id)
    if kalmar and kalmar.init_segment:
        try:
            await websocket.send_bytes(kalmar.init_segment)
            print(f"[WS] Sent init segment ({len(kalmar.init_segment)} bytes)")
        except Exception as e:
            print("Init send error:", e)

    try:
        while True:
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        pass

    finally:
        async with clients_lock:
            clients.get(kalmar_id, set()).discard(websocket)

        print(f"[WS] Viewer disconnected from {kalmar_id}")





# ================= REST API Endpoints =================
@app.post("/init")
def init_bulk(configs: List[KalmarConfig], background: BackgroundTasks):
    started = []
    for c in configs:
        try:
            print(f"[INIT] Starting {c.kalmarId}")

            if c.kalmarId in kalmar_sessions:
                print(f"[INIT] Restarting existing session for {c.kalmarId}")
                kalmar_sessions[c.kalmarId].stop()
                time.sleep(1)

            session = KalmarSession(c.kalmarId)
            kalmar_sessions[c.kalmarId] = session
            session.start_ffmpeg(rtsp_url=c.rtspUrl)

            started.append(c.kalmarId)
            print(f"[INIT] ✅ Started {c.kalmarId}")

        except Exception as e:
            print(f"[INIT] ❌ Error starting {c.kalmarId}: {e}")
            traceback.print_exc()

    return {"status": "ok", "started": started}


@app.get("/active_kalmars")
def get_active_kalmars():
    # if WEBSOCKET_ENABLED:
        active = list(workers)
        return {"active_kalmars": active, "count": len(workers)}
    # return {"active_kalmars": [], "count": 0}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "websocket": WEBSOCKET_ENABLED,
        "active_streams": len(kalmar_sessions),
        "ocr_handler_ready": True,
        "timestamp": datetime.now().isoformat(),
    }




# ================= MAIN =================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Kalmar FFmpeg Service (WITH TIMESTAMP-MATCHED LOCATION TRACKING)")
    print("=" * 60)
    print("📹 Features:")
    print(f"   Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"   Display FPS: 30 target")
    print(f"   WebSocket FPS: 30 target")
    print(f"   Model FPS: {TARGET_MODEL_FPS}")
    print(f"   Device: {DEVICE}")
    print(f"   Mode: {'VIDEO FILE TESTING' if USE_VIDEO_FILE else 'RTSP CAMERA'}")
    if USE_VIDEO_FILE:
        print(f"   Video: {VIDEO_FILE_PATH}")
    # print(f"   Placed API: {PLACED_API_URL}")
    print("📡 Endpoints:")
    print("  POST /init           - Start RTSP streams")
    print("  WS   /ws-ingest/{id} - Ingest H.264 + location stream")
    print("  WS   /ws/{id}        - Broadcast frames")
    print("  GET  /active_kalmars - List active streams")
    print("  GET  /health         - Service health")
    print("⌨️ Window Controls:")
    print("  Press 'q' - Quit all streams")
    print("  Press 's' - Save screenshot")
    print("  Press 'd' - Debug info")
    print("=" * 60)
    # ================= AUTO START SINGLE RTSP =================
    print(f"🎬 Auto-starting Kalmar {SINGLE_KALMAR_ID}")
    session = KalmarSession(SINGLE_KALMAR_ID)
    kalmar_sessions[SINGLE_KALMAR_ID] = session
    
    # 🔧 Start with video file or RTSP based on config
    if USE_VIDEO_FILE:
        session.start_ffmpeg(rtsp_url=None)  # Will use video file
    else:
        session.start_ffmpeg(rtsp_url=SINGLE_RTSP_URL)
       
    # ================= START LOCATION PULLER =================
    location_fetcher = KalmarLocationFetcher(
        kalmar_id=SINGLE_KALMAR_ID,
        ocr_handler=ocr_handler
    )
    location_fetcher.start()



    # Start model thread
    model_thread = threading.Thread(target=model_loop, daemon=True)
    model_thread.start()
    print("🧠 Model loop started")

    # Run FastAPI in a background thread
    def run_fastapi():
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=8001,
            ws_ping_timeout=60,
            ws_ping_interval=30,
            log_level="info",
        )
        server = uvicorn.Server(config)

        global ASYNC_LOOP
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ASYNC_LOOP = loop
        loop.run_until_complete(server.serve())

    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()

    print("\n⏳ Waiting 2 seconds for FastAPI server to start...")
    time.sleep(2)
    print("✅ FastAPI server is running on http://localhost:8001")

    # ⚡ CRITICAL: Run OpenCV display in the MAIN THREAD
    print(" 🖥️ Starting OpenCV display in MAIN THREAD...")
    try:
        if not RUN_AS_SERVICE and not HEADLESS:
         main_display_loop()
        else:
         print("🚫 Headless / service mode: OpenCV display disabled")
    except KeyboardInterrupt:
        print("\n⌨️ Keyboard interrupt received")
    except Exception as e:
        print(f"\n❌ Error in display loop: {e}")
        traceback.print_exc()
    finally:
        if not HEADLESS:
         cv2.destroyAllWindows()
         print("✅ Clean shutdown completed")

         
         
         
         
    # ==================================================
    # ⏸️ KEEP SERVICE ALIVE (THIS IS WHAT YOU ADD)
    # ==================================================
    if RUN_AS_SERVICE:
        print("🔄 Service mode active → blocking main thread")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("🛑 Service stopped")
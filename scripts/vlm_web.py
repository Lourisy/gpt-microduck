#!/usr/bin/env python3
"""Minimal browser controller for Microduck's official ONNX skills.

The simulator keeps the ONNX loop local at 50 Hz.  The browser only sends task
text and polls state/video; GPT is called once per skill boundary.
"""

from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import os
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# The web server is headless by default. Users with a desktop viewer can
# override this before launch, e.g. MUJOCO_GL=glfw.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image

try:
    from infer_policy import (
        DEFAULT_POSE, PolicyInference,
        load_bam_model, load_mujoco_with_bam,
    )
except ModuleNotFoundError:  # import scripts.vlm_web from the repository root
    from scripts.infer_policy import (
        DEFAULT_POSE, PolicyInference,
        load_bam_model, load_mujoco_with_bam,
    )

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ROOT / "official_policies"
HTML = Path(__file__).with_name("vlm_web.html")
DEFAULT_SCENE = ROOT / "src/mjlab_microduck/robot/microduck/scene_ball.xml"

# The head camera is mounted sideways in the MJCF. Render it in portrait and
# rotate the result so the browser/VLM receive a normal landscape frame.
HEAD_WIDTH, HEAD_HEIGHT = 640, 360
THIRD_WIDTH, THIRD_HEIGHT = 640, 360
HEAD_CAMERA_FOVY_DEG = 65.0


def config():
    try:
        from vlm_local_config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
    except ImportError as exc:
        try:
            from scripts.vlm_local_config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
        except ImportError:
            raise RuntimeError("create scripts/vlm_local_config.py first") from exc
    return {
        "key": os.getenv("OPENAI_API_KEY", OPENAI_API_KEY),
        "base": os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL).rstrip("/"),
        "model": os.getenv("OPENAI_MODEL", OPENAI_MODEL),
    }


SYSTEM_PROMPT = """You control a Microduck robot through one skill per turn.
Choose only one of: walking, stand, sitstand, ground_pick, kick_left, kick_right,
roulade, done. The robot uses official ONNX controllers; you do not output joint
actions. Walking arguments are vx, vy in m/s, wz in rad/s, and duration_s.
Moving forward, moving left, and turning left (counterclockwise) correspond to
the positive directions of vx, vy, and wz, respectively. You can flexibly
control the position by moving, turning, and using different durations.
It is recommended to set vx or vy to a constant value of 0.3 m/s,
and wz to a constant value of 1.5 rad/s.
It is recommended that the duration_s be no less than 0.3 seconds
and no more than 2 seconds each time, and since the turning angular
velocity is relatively fast, the corresponding duration_s for turning
skills should be slightly lower.
Walking duration controls how far the robot moves.
so inspect the next image and result before deciding the next skill.
ground_pick, kick_left, kick_right, roulade, sitstand, and stand end from their
local controller/state/fixed timer; do not provide duration_s for those skills. sitstand
requires arguments {"mode":"sit"} or {"mode":"stand"}; use one call for each
posture transition. For example, sitting is
{"skill":"sitstand","arguments":{"mode":"sit"}} and standing is
{"skill":"sitstand","arguments":{"mode":"stand"}}.
Walking stops after duration; the other skills stop when their local completion
condition is reached. If a local condition does not arrive, the server reports
timeout after its safety limit. Decide whether
the previous skill completed from the new image and state. Return JSON only:
{"text":"short observation","skill":"...","arguments":{...}}.
Use done when the user's task is complete or cannot be completed."""

SITSTAND_WAIT_S = 3.0
# Skill durations use simulation time; this watchdog uses wall-clock time.
SKILL_TIMEOUT_S = 10.0
FRAME_INTERVAL_S = 1.0 / 30.0


def response_text(payload: dict) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    chunks: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []) if isinstance(item, dict) else []:
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                chunks.append(content["text"])
    return "\n".join(chunks)


# call_vlm() output for example:
# {
#     "text": "向前走",
#     "skill": "walking",
#     "arguments": {
#         "vx": 0.3,
#         "vy": 0.0,
#         "wz": 0.0,
#         "duration_s": 2.0
#     }
# }

def call_vlm(text: str, image: bytes, context: dict, fake: bool = False) -> dict:
    if fake:
        # A deterministic smoke-test policy: execute the user's requested text as
        # a short forward walk, then finish on the next turn.
        if context.get("last_result") is None:
            return {"text": "fake VLM walk", "skill": "walking",
                    "arguments": {"vx": 0.3, "vy": 0.0, "wz": 0.0, "duration_s": 1.0}}
        return {"text": "fake VLM done", "skill": "done", "arguments": {"summary": "smoke test"}}

    cfg = config()
    data_url = "data:image/jpeg;base64," + base64.b64encode(image).decode()
    prompt = SYSTEM_PROMPT + "\n\nUser task: " + text + "\nCurrent robot context:\n" + json.dumps(context, ensure_ascii=False)
    body = {
        "model": cfg["model"],
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": data_url},
        ]}],
    }
    request = urllib.request.Request(
        cfg["base"] + "/responses",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + cfg["key"]},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        raw = json.loads(response.read())
    answer = response_text(raw).strip()
    if answer.startswith("```"):
        answer = answer.strip("`").removeprefix("json").strip()
    return json.loads(answer)


class Sim:
    def __init__(self, scene: str, fake_vlm: bool):
        self.lock = threading.RLock()
        self.fake_vlm = fake_vlm
        bam = load_bam_model(200.0, 7.4, None)
        self.model, self.data, self.bam, _ = load_mujoco_with_bam(scene, bam, 0.005, 0.1, 6.0)
        self.model.vis.global_.offheight = max(HEAD_WIDTH, THIRD_HEIGHT)
        self.model.vis.global_.offwidth = max(HEAD_HEIGHT, THIRD_WIDTH)
        # The XML head camera is mounted backwards. Keep the same correction as
        # mjlab_microduck.sim.camera.Camera, otherwise the image is just the
        # camera housing/near geometry (a gray circle).
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
        if camera_id < 0:
            raise RuntimeError("scene has no head_camera")
        turn = np.array([0.0, 1.0, 0.0, 0.0])
        fixed = np.zeros(4)
        mujoco.mju_mulQuat(fixed, self.model.cam_quat[camera_id], turn)
        self.model.cam_quat[camera_id] = fixed
        self.model.cam_fovy[camera_id] = HEAD_CAMERA_FOVY_DEG
        self.policy = PolicyInference(
            self.model, self.data, bam_ctrl=self.bam, new_cmd_obs=True,
            use_projected_gravity=True,
            walking_onnx_path=str(ROOT / "my_onnx/velocity_rough_model_49999.onnx"),
            standing_onnx_path=str(POLICIES / "alpha_stand.onnx"),
            ground_pick_onnx_path=str(POLICIES / "alpha_ground_pick.onnx"),
            sitstand_onnx_path=str(POLICIES / "alpha_sitstand.onnx"),
            kick_left_onnx_path=str(POLICIES / "ball_kick_left.onnx"),
            kick_right_onnx_path=str(POLICIES / "ball_kick_right.onnx"),
            roulade_onnx_path=str(POLICIES / "roulade.onnx"),
        )
        self._reset()
        self.frame = b""
        self.frame_time = -1.0
        self.third_frame = b""
        self.third_camera = mujoco.MjvCamera()
        self.third_camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.third_camera.trackbodyid = self.policy.trunk_base_id
        self.third_camera.distance = 0.95
        self.third_camera.azimuth = 135.0
        self.third_camera.elevation = -18.0
        self.third_camera.lookat[:] = [0.0, 0.0, 0.10]
        self.task_text = ""
        self.history: list[dict] = []
        self.current: dict | None = None
        self.shutdown_requested = threading.Event()
        self.task_cancel = threading.Event()
        self.task_thread: threading.Thread | None = None
        self.loop_thread = threading.Thread(target=self._loop, daemon=True)
        self.render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self.loop_thread.start()
        self.render_thread.start()

    def _reset(self):
        with self.lock:
            free = self.model.joint("trunk_base_freejoint").id
            qa = int(self.model.jnt_qposadr[free])
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[qa:qa + 7] = [0, 0, 0.125, 1, 0, 0, 0]
            self.data.qpos[self.policy.joint_qpos_indices] = DEFAULT_POSE
            self.bam.reset(self.data.qpos)
            self.policy.last_action[:] = 0
            self.policy.set_position_targets(DEFAULT_POSE)
            # Keep the robot upright while waiting for the first VLM command.
            # Without a policy pass in the idle state, the default pose falls
            # under gravity before the browser requests its next frame.
            self.policy.set_vel_cmd(0.0, 0.0, 0.0)
            mujoco.mj_forward(self.model, self.data)

    def snapshot(self) -> dict:
        with self.lock:
            g = self.policy.get_projected_gravity()
            upright = float(-g[2])
            return {
                "upright_score": round(upright, 3),
                "policy": self.policy.current_policy,
                "command": self.policy.command[:3].round(3).tolist(),
            }

    def image(self, after: float = -1.0, cancel=None) -> bytes:
        deadline = time.monotonic() + 5.0
        while (not self.frame or self.frame_time < after) and time.monotonic() < deadline:
            if cancel is not None and cancel.is_set():
                return b""
            time.sleep(0.02)
        if not self.frame or self.frame_time < after:
            raise RuntimeError("camera frame is not ready")
        return self.frame

    def third_image(self) -> bytes:
        deadline = time.monotonic() + 5.0
        while not self.third_frame and time.monotonic() < deadline:
            time.sleep(0.02)
        if not self.third_frame:
            raise RuntimeError("third-person frame is not ready")
        return self.third_frame

    def orbit_camera(self, azimuth_delta=0.0, elevation_delta=0.0, zoom_delta=0.0):
        with self.lock:
            self.third_camera.azimuth = (self.third_camera.azimuth + float(azimuth_delta)) % 360.0
            self.third_camera.elevation = float(np.clip(
                self.third_camera.elevation + float(elevation_delta), -80.0, 35.0
            ))
            self.third_camera.distance = float(np.clip(
                self.third_camera.distance + float(zoom_delta), 0.35, 3.0
            ))

    def reset_camera(self):
        with self.lock:
            self.third_camera.azimuth = 135.0
            self.third_camera.elevation = -18.0
            self.third_camera.distance = 0.95

    def submit(self, text: str):
        with self.lock:
            if (self.shutdown_requested.is_set() or not text
                    or (self.task_thread and self.task_thread.is_alive()
                            and not self.task_cancel.is_set())):
                return False
            self.task_text = text
            self.history = []
            self.task_cancel = threading.Event()
            self._set_status("requesting_vlm", "正在获取观测并请求 GPT…", {})
            self.task_thread = threading.Thread(target=self._task, args=(text, self.task_cancel), daemon=True)
            self.task_thread.start()
        return True

    def stop(self):
        with self.lock:
            self.task_cancel.set()
            self._hold()
            self._set_status("stopped", "已停止；可以发送新任务。", {})

    def _hold(self):
        """Called under lock: cancel active motion, preserving a seated posture."""
        self.policy.ground_pick_mode = False
        self.policy.behavior_mode = None
        if self.policy.current_policy != "sit":
            self.policy.sit_mode = False
            self.policy.set_vel_cmd(0, 0, 0)

    def shutdown(self):
        # Stop the task first so image waits and skill waits wake up, then stop
        # the two long-running simulation threads. A VLM HTTP request cannot be
        # interrupted by urllib, so only wait briefly for that daemon worker.
        self.stop()
        self.shutdown_requested.set()
        task_thread = self.task_thread
        if task_thread and task_thread is not threading.current_thread():
            task_thread.join(timeout=1.0)
        self.loop_thread.join(timeout=2.0)
        self.render_thread.join(timeout=2.0)

    def _cancelled(self, cancel: threading.Event) -> bool:
        """Return whether this task or the whole simulator is stopping."""
        return cancel.is_set() or self.shutdown_requested.is_set()

    def _task(self, text: str, cancel: threading.Event):
        context = {"last_result": None}
        while not self._cancelled(cancel):
            try:
                with self.lock:
                    if self._cancelled(cancel):
                        return
                    self._set_status("requesting_vlm", "正在获取观测并请求 GPT…", {})
                # Use the latest completed render. Rendering is deliberately
                # decoupled from physics and can lag by a few frames; waiting
                # for an exact simulation timestamp would reintroduce latency.
                previous_time = self.frame_time
                frame = self.image(after=previous_time+1e-6,cancel=cancel)
                if self._cancelled(cancel):
                    return
                decision = call_vlm(text, frame, {**context, **self.snapshot()}, self.fake_vlm)
                if self._cancelled(cancel):
                    return
                with self.lock:
                    # A stopped HTTP request may finish later. It must never
                    # start a skill or overwrite the next task's state.
                    if self._cancelled(cancel):
                        return
                    if decision["skill"] == "done":
                        self._set_status("done", decision.get("text", "done"), decision)
                        return
                    self._validate(decision)
                    self._start_skill(decision)
                result = self._wait_skill(cancel)
                if self._cancelled(cancel):
                    return
                with self.lock:
                    context = {
                        "last_skill": decision, "last_result": result,
                        "event": result["result"], **self.snapshot(),
                    }
                    self.history.append(context)
            except Exception as exc:
                with self.lock:
                    if not self._cancelled(cancel):
                        self._hold()
                        self._set_status("error", str(exc), {})
                return

    @staticmethod
    def _validate(decision: dict):
        skills = {"walking", "stand", "sitstand", "ground_pick", "kick_left", "kick_right", "roulade", "done"}
        if decision.get("skill") not in skills or not isinstance(decision.get("arguments", {}), dict):
            raise ValueError("invalid VLM skill JSON")
        args = decision["arguments"]
        if decision["skill"] == "walking":
            for key, lo, hi in (("vx", -.4, .4), ("vy", -.4, .4), ("wz", -2.0, 2.0)):
                if not lo <= float(args.get(key, 0)) <= hi:
                    raise ValueError(f"walking {key} out of range")
        if decision["skill"] == "walking":
            if not 0.01 <= float(args.get("duration_s", 0.0)) <= SKILL_TIMEOUT_S:
                raise ValueError(
                    f"walking duration_s must be between 0.1 and {SKILL_TIMEOUT_S:g} seconds"
                )
        if decision["skill"] == "sitstand" and args.get("mode") not in {"sit", "stand"}:
            raise ValueError('sitstand arguments must include mode: "sit" or "stand"')

    def _start_skill(self, decision: dict):
        name, args = decision["skill"], decision["arguments"]
        with self.lock:
            self.policy.ground_pick_mode = False
            self.policy.behavior_mode = None
            if name == "walking":
                self.policy.sit_mode = False
                self.policy.set_vel_cmd(float(args.get("vx", 0)), float(args.get("vy", 0)), float(args.get("wz", 0)))
            elif name == "stand":
                self.policy.sit_mode = False
                self.policy.current_policy = "standing"
                self.policy.ort_session = self.policy.standing_session
                self.policy.set_vel_cmd(0, 0, 0)
            elif name == "sitstand":
                self.policy.current_policy = "sit"
                self.policy.ort_session = self.policy.sit_session
                self.policy.sit_mode = args.get("mode", "stand") == "sit"
                self.policy._update_command()
            elif name == "ground_pick":
                self.policy.trigger_ground_pick()
            elif name in {"kick_left", "kick_right", "roulade"}:
                self.policy.trigger_behavior(name)
            self.current = {**decision, "started": time.time(), "started_sim": self.data.time, "status": "running"}

    def _wait_skill(self, cancel: threading.Event) -> dict:
        with self.lock:
            if self._cancelled(cancel):
                return {"result": "stopped", "message": "operator stopped"}
            skill = self.current["skill"]
            duration = float(self.current["arguments"].get("duration_s", 0.0))
            started_sim = self.current["started_sim"]
        start = time.monotonic()

        while not self._cancelled(cancel):
            with self.lock:
                if self._cancelled(cancel):
                    break
                elapsed = self.data.time - started_sim
                wall_elapsed = time.monotonic() - start
                complete = False
                if skill == "walking":
                    complete = elapsed >= duration
                elif skill == "ground_pick":
                    complete = not self.policy.ground_pick_mode
                elif skill in {"kick_left", "kick_right", "roulade"}:
                    complete = self.policy.behavior_mode is None
                elif skill in {"sitstand", "stand"}:
                    complete = elapsed >= SITSTAND_WAIT_S
                if complete or wall_elapsed >= SKILL_TIMEOUT_S:
                    result = "completed" if complete else "timeout"
                    self.current["status"] = result
                    self._hold()
                    return {
                        "result": result,
                        "message": f"skill {result}",
                        "sim_s": round(elapsed, 3), "wall_s": round(wall_elapsed, 3),
                    }
            cancel.wait(0.02)
        return {"result": "stopped", "message": "operator stopped"}

    def _set_status(self, status: str, message: str, details: dict):
        with self.lock:
            self.current = {"status": status, "message": message, **details}

    def _loop(self):
        last = time.perf_counter()
        while not self.shutdown_requested.is_set():
            now = time.perf_counter()
            if now - last < 0.02:
                time.sleep(0.001)
                continue
            last = now
            with self.lock:
                running = bool(self.current and self.current.get("status") == "running")
                if running or self.policy.ground_pick_mode or self.policy.behavior_mode is not None:
                    self.policy.update_ground_pick_phase(0.02)
                    self.policy.update_behavior(0.02)
                # A completed skill leaves the controller in a waiting state
                # while GPT is called again. Stop walking and keep standing so
                # the robot remains under the last safe controller target
                if not running and self.policy.current_policy == "walking":
                    self.policy.set_vel_cmd(0.0, 0.0, 0.0)
                self.policy.apply_action(self.policy.infer())
                if self.bam:
                    for _ in range(4):
                        self.bam.update()
                        mujoco.mj_step(self.model, self.data)
    def _render_loop(self):
        # EGL calls can take hundreds of milliseconds. Render a private copy;
        # never hold the physics/HTTP lock during GPU work or JPEG encoding.
        render_data = mujoco.MjData(self.model)
        # Use independent renderers because the head camera needs the inverse
        # (portrait) raster dimensions before its quarter-turn correction.
        with (
            mujoco.Renderer(self.model, height=HEAD_WIDTH, width=HEAD_HEIGHT) as head_renderer,
            mujoco.Renderer(self.model, height=THIRD_HEIGHT, width=THIRD_WIDTH) as third_renderer,
        ):
            while not self.shutdown_requested.is_set():
                started = time.monotonic()
                with self.lock:
                    mujoco.mj_copyData(render_data, self.model, self.data)
                    third_camera = copy.copy(self.third_camera)
                head_renderer.update_scene(render_data, camera="head_camera")
                head_rgb = head_renderer.render().copy()
                third_renderer.update_scene(render_data, camera=third_camera)
                third_rgb = third_renderer.render().copy()
                # The Microduck head camera is mounted a quarter turn sideways.
                # The official camera pipeline applies the same clockwise 90°
                # correction before consumers see the frame. Keep the MuJoCo
                # camera convention unchanged and rotate only this observation.
                head_rgb = np.rot90(head_rgb, k=3).copy()
                head_out = io.BytesIO()
                Image.fromarray(head_rgb).save(head_out, format="JPEG", quality=78)
                third_out = io.BytesIO()
                Image.fromarray(third_rgb).save(third_out, format="JPEG", quality=82)
                self.frame = head_out.getvalue()
                self.frame_time = render_data.time
                self.third_frame = third_out.getvalue()
                self.shutdown_requested.wait(max(0, FRAME_INTERVAL_S - (time.monotonic() - started)))

    def state(self) -> dict:
        with self.lock:
            return {"task": self.task_text, "current": self.current, "history": self.history[-5:], **self.snapshot()}


class Handler(BaseHTTPRequestHandler):
    sim: Sim

    def _send(self, code: int, body: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/":
            return self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
        if self.path.startswith("/api/frame_stream"):
            return self._stream(self.sim.image)
        if self.path.startswith("/api/third_stream"):
            return self._stream(self.sim.third_image)
        if self.path.startswith("/api/state"):
            return self._send(200, json.dumps(self.sim.state()).encode(), "application/json")
        if self.path.startswith("/api/frame"):
            return self._send(200, self.sim.frame or self.sim.image(), "image/jpeg")
        if self.path.startswith("/api/third_frame"):
            return self._send(200, self.sim.third_frame or self.sim.third_image(), "image/jpeg")
        self._send(404, b"not found", "text/plain")

    def _stream(self, get_frame):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while not self.sim.shutdown_requested.is_set():
                frame = get_frame()
                self.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(frame)).encode()
                    + b"\r\n\r\n" + frame + b"\r\n"
                )
                self.wfile.flush()
                time.sleep(FRAME_INTERVAL_S)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):  # noqa: N802
        size = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(size) or b"{}")
        if self.path == "/api/task":
            ok = self.sim.submit(str(body.get("text", "")).strip())
            message = "任务已提交" if ok else "已有任务正在执行，请先停止"
            return self._send(200, json.dumps({"accepted": ok, "message": message}, ensure_ascii=False).encode(), "application/json")
        if self.path == "/api/stop":
            self.sim.stop()
            return self._send(200, b'{"stopped":true}', "application/json")
        if self.path == "/api/camera":
            self.sim.orbit_camera(
                body.get("azimuth", 0), body.get("elevation", 0), body.get("zoom", 0)
            )
            return self._send(200, b'{"ok":true}', "application/json")
        if self.path == "/api/camera/reset":
            self.sim.reset_camera()
            return self._send(200, b'{"ok":true}', "application/json")
        self._send(404, b"not found", "text/plain")

    def log_message(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--scene", default=str(DEFAULT_SCENE),
        help="MuJoCo scene XML (default: scene_ball.xml)",
    )
    parser.add_argument("--fake-vlm", action="store_true", help="run without calling GPT")
    args = parser.parse_args()
    # import pdb; pdb.set_trace();
    sim = Sim(args.scene, args.fake_vlm)
    Handler.sim = sim
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Microduck VLM web: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
        sim.shutdown()


if __name__ == "__main__":
    main()

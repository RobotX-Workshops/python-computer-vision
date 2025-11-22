# Remote Camera Streaming Guide

This guide explains how to expose a physical webcam to the workshop exercises when you cannot pass `/dev/video0` directly into a container (for example on macOS or Windows Docker Desktop). The approach is the same on every platform:

1. Run a host-side tool that captures the webcam and publishes it over the network (UDP, RTSP, HTTP, or HLS).
2. Point OpenCV inside the container (or on another machine) at the network stream instead of a local camera index.

The exercises now accept a `--video-source` argument (or prompt you interactively) so you can choose between `0` (local webcam) and a network URL such as `udp://host.docker.internal:5000`.

## Quick Reference

- **Default webcam:** `python realtime_face_detection.py --video-source 0`
- **UDP stream:** `python realtime_face_detection.py --video-source udp://host.docker.internal:5000`
- **HTTP/MJPEG stream:** `python realtime_face_detection.py --video-source http://host.docker.internal:8080/video`
- **Environment variable:** set `VIDEO_SOURCE=udp://host.docker.internal:5000` before launching the script.

`host.docker.internal` resolves to the host machine from Docker containers on macOS and Windows. On native Linux hosts you can use `localhost` or the host IP directly.

## 1. Host Streaming Setup

### macOS

1. **List available AVFoundation devices:**

   ```bash
   ffmpeg -f avfoundation -list_devices true -i ""
   ```

   Note the video device index (usually `0`).

2. **Start a UDP stream at 1280×720 30 FPS:**

   ```bash
   ffmpeg -f avfoundation \
          -framerate 30 \
          -video_size 1280x720 \
          -pixel_format uyvy422 \
          -i "0:none" \
          -f mpegts udp://127.0.0.1:5000
   ```

   Replace `0` if your camera uses a different index. Leave the terminal running; it keeps the stream alive.

3. **Optional – HLS for browsers:**

   ```bash
   mkdir -p ~/camera-hls
   cd ~/camera-hls
   ffmpeg -f avfoundation \
          -framerate 30 \
          -video_size 1280x720 \
          -pixel_format uyvy422 \
          -i "0:none" \
          -preset ultrafast \
          -f hls -hls_time 2 -hls_list_size 4 -hls_flags delete_segments \
          stream.m3u8
   ```

   Serve the files with `python -m http.server 8080` and open `http://localhost:8080/stream.m3u8` in Safari.

### Windows (PowerShell)

1. Install FFmpeg (e.g. via [Chocolatey](https://chocolatey.org/) `choco install ffmpeg`).
2. List DirectShow devices:

   ```powershell
   ffmpeg -list_devices true -f dshow -i dummy
   ```

3. Start a UDP stream:

   ```powershell
   ffmpeg -f dshow \
          -framerate 30 \
          -video_size 1280x720 \
          -i "video=Integrated Camera" \
          -f mpegts udp://127.0.0.1:5000
   ```

   Change `Integrated Camera` to match your device name exactly.

4. Optional HLS output (same pattern as macOS but using `-f dshow`).

### Linux

1. Enumerate V4L2 devices (`/dev/video*`).
2. Start streaming with FFmpeg or GStreamer:

   ```bash
   ffmpeg -f v4l2 \
          -framerate 30 \
          -video_size 1280x720 \
          -i /dev/video0 \
          -f mpegts udp://127.0.0.1:5000
   ```

3. If the container runs on the same host, expose the UDP port with `socat` or by binding to `0.0.0.0`:

   ```bash
   ffmpeg ... -f mpegts udp://0.0.0.0:5000
   ```

   Inside the container use `udp://host.docker.internal:5000` (Docker Desktop) or `udp://<host-ip>:5000` (native Docker).

## 2. Using the Stream Inside the Container

### Configure the devcontainer (optional)

Add the following to `.devcontainer/devcontainer.json` if you want the UDP port available automatically:

```jsonc
"forwardPorts": [5000]
```

Rebuild or reopen the container after editing the configuration.

### Launching the exercises

Each script now checks the `VIDEO_SOURCE` environment variable or accepts a `--video-source` flag.

Examples:

```bash
# Inside the container
export VIDEO_SOURCE=udp://host.docker.internal:5000
python 02-face-detection/realtime_face_detection.py

# Override explicitly
python 02-face-detection/realtime_face_detection.py --video-source udp://host.docker.internal:5000
python 02-face-detection/camera_face_capture.py --video-source udp://host.docker.internal:5000
python 02-face-detection/simple_camera_demo.py --video-source udp://host.docker.internal:5000
```

If you are using the HLS option, pass the HTTP URL instead (for example `http://host.docker.internal:8080/stream.m3u8`).

## 3. Testing the Stream

1. **Verify on the host:** open `ffplay udp://127.0.0.1:5000` or use VLC to confirm the stream works before involving containers.
2. **Verify inside the container:**

   ```bash
   ffprobe udp://host.docker.internal:5000
   ```

   A successful probe prints stream details (codec, resolution, FPS).

3. **Fallback plan:** if network streaming is not available, the scripts gracefully fall back to camera index `0`, so you can still run them outside containers.

## 4. Troubleshooting

- **No frames received:** check that FFmpeg is still running and that the UDP port is forwarded to the container.
- **High latency:** reduce resolution or frame rate (`-video_size 640x480`, `-framerate 15`).
- **Browser playback issues:** prefer HLS (`.m3u8`) for Safari/Edge, or MJPEG/RTSP for Chrome.
- **`host.docker.internal` not resolving on Linux:** replace it with the host IP (for example `172.17.0.1` or the address returned by `ip addr show docker0`).

With this setup you can run all the camera-based exercises in environments that do not grant direct access to `/dev/video0`, including macOS and Windows devcontainers, remote servers, or cloud VMs.

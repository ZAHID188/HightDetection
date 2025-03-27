# HightDetection


# Accurately Measuring The Height Of The Train Surafce

## Several approaches:
- camera or webcam
- LiDAR or depth sensors (like Intel RealSense)
- Ultrasonic sensors
## Chosen approach is using Camera Calibration
1.	OpenCV to correct for lens distortion
2.	A reference object of known dimensions to establish scale.

### Requiremets
- python version 3.8.20(conda activate myenv)

- Check for camera devices:
    - ` ls -l /dev/video* `
    -    ```v4l2-ctl --list-devices```
    - ```v4l2-ctl --device=/dev/videoX --all```
    - `sudo apt install cheese` then `run cheese`

# Live-Bottle-Detection
This is a Python script that uses several libraries to perform real-time bottle detection using a pre-trained deep learning model. Here's a breakdown of what each part of the code does:

The torch and cv2 (OpenCV) libraries are imported for deep learning inference and image processing. numpy is used for numerical operations, and other utility modules are imported to handle video streams and model loading.

A custom-trained YOLOv5 model is loaded from the best.pt file. This model is trained specifically to detect bottles in images or video.

The script sets up the input source. It can either use a live webcam feed (typically device index 0) or a path to an image or video file.

For each frame captured from the input source, the script uses the YOLOv5 model to perform object detection.

If the model detects a bottle, it returns the bounding box coordinates, confidence score, and class label ("bottle").

The detection results are drawn on the frame using colored rectangles and labeled with the class name and confidence percentage.

The annotated frame is either displayed live on the screen using OpenCV’s imshow() function or saved as an output file, depending on the script settings.

The detection loop continues to run in real time, processing frame-by-frame until the user presses a specific key (e.g., 'q') to quit.

Finally, all resources like camera access and display windows are properly released and closed using cv2.destroyAllWindows().

In summary, this script allows the user to run a real-time bottle detection system using a YOLOv5 model. It captures video, detects bottles, highlights them with bounding boxes, and optionally displays or saves the results.


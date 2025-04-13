import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import mediapipe as mp
import math

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Function to calculate Euclidean distance
def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def calculate_golden_ratio(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("❌ Error: Could not load image!")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(img_rgb)

    if not result.multi_face_landmarks:
        print("❌ No face detected!")
        return

    h, w, _ = img.shape
    landmarks = result.multi_face_landmarks[0]

    points = {}
    for id in [10, 152, 33, 263]:
        lm = landmarks.landmark[id]
        points[id] = (int(lm.x * w), int(lm.y * h))
        cv2.circle(img, points[id], 3, (0, 255, 0), -1)

    face_height = distance(points[10], points[152])
    face_width = distance(points[33], points[263])

    golden_ratio = 1.618
    ratio = face_height / face_width
    percentage = min(100, (1 - abs(ratio - golden_ratio) / golden_ratio) * 100)

    print(f"\n📊 Analysis Results:")
    print(f"   • Face Height: {face_height:.2f} px")
    print(f"   • Face Width : {face_width:.2f} px")
    print(f"   • Golden Ratio Estimate: {ratio:.2f}")
    print(f"   • Symmetry Match: {percentage:.2f}%\n")

    if percentage > 90:
        print("✨ Exceptional symmetry — an outstanding match to the golden ratio.")
    elif percentage > 80:
        print("✅ Very well-balanced facial proportions.")
    elif percentage > 70:
        print("👍 Above average facial symmetry.")
    elif percentage > 50:
        print("ℹ️ Average proportional balance.")
    else:
        print("🌱 Unique facial structure — beauty isn't defined by numbers.")

    cv2.imshow('Facial Analysis Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Input image path
image_path = input("📂 Enter the path to your image: ").strip('"')
calculate_golden_ratio(image_path)

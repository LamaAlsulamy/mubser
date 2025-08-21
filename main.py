import cv2
import streamlit as st
import numpy as np

# إعداد واجهة ستريملت
st.set_page_config(page_title="Face Detection Viewer", layout="wide")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("logo1.jpeg", width=300)

# دالة لاكتشاف الكاميرات المتاحة
@st.cache_resource
def get_available_cameras(max_devices=5):
    available = []
    for index in range(max_devices):
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)  # مهم للماك
        if cap is not None and cap.isOpened():
            available.append(index)
            cap.release()
    return available

# تحميل كاشف الوجوه
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# العنوان
st.title("📷 Face Detection | Thermal vs Normal")

# الحصول على الكاميرات المتاحة
camera_indices = get_available_cameras()
if not camera_indices:
    st.error("⚠️ No cameras found. تأكدي من إعطاء صلاحية الكاميرا للتيرمنال أو VS Code.")
    st.stop()

# اختيار الكاميرا
camera_id = st.selectbox("Choose camera", camera_indices, index=0)
run = st.checkbox("Run Camera", value=False)

# تقسيم الشاشة لنصين
col1, col2 = st.columns(2)
with col1:
    st.subheader("🌡️ Thermal View")
    thermal_window = st.image([], use_container_width=True)

with col2:
    st.subheader("🙂 Normal View")
    normal_window = st.image([])

# تشغيل الكاميرا لو التفعيل موجود
if run:
    cap = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION)  # مهم للماك

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to read from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # كشف الوجوه
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(95, 95)  # فلترة الوجوه الصغيرة
        )

        # عرض الحرارة
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        for (x, y, w, h) in faces:
            cv2.rectangle(thermal, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # العرض العادي
        annotated = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated, "Tasneem", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # تحويل الألوان لـ RGB عشان ستريملت
        thermal_rgb = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        # تحديث العرض
        thermal_window.image(thermal_rgb)
        normal_window.image(annotated_rgb)

    cap.release()

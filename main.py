import cv2

def overlay_png(background, overlay, x, y):
    h, w = overlay.shape[:2]
    bh, bw = background.shape[:2]

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, bw), min(y + h, bh)
    ox1, oy1 = x1 - x, y1 -y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 -y1)

    if x2 <= x1 or y2 <= y1:
        return
    
    overlay_crop = overlay[oy1:oy2, ox1: ox2]
    bg_crop = background[y1:y2, x1:x2]

    alpha = overlay_crop[:, :, 3:4] /255.0
    overlay_rgb = overlay_crop[:, :, :3]

    background[y1:y2, x1:x2] = (overlay_rgb * alpha + bg_crop * (1 - alpha)).astype('uint8')


cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

mustache = cv2.imread("curly-black-mustache-free-png.png", cv2.IMREAD_UNCHANGED)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))
    
    for (x, y, w, h) in faces:
        mw = int(w * 0.8)
        mh = int (mw * mustache.shape[0] / mustache.shape[1])
        resized_mustache = cv2.resize(mustache, (mw, mh))

        mx = x + (w - mw) // 2
        my = y + int(h * 0.4)
        
        overlay_png(frame, resized_mustache, mx, my)

    cv2.imshow("Mustache Cam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
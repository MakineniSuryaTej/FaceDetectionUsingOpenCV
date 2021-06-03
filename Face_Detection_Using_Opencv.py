import cv2

face_cascade = cv2.CascadeClassifier('OtherResources/haarcascade_frontalface_default.xml')  # Loading xml file for faces
eye_cascade = cv2.CascadeClassifier('OtherResources/haarcascade_eye.xml')  # Loading xml for eyes
cap = cv2.VideoCapture(1)  # Parameter 1 is for WebCam and 0 is for Integrated Camera
while True:  # Infinite Loop for continuous video capture results
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converting img to gray image
    faces = face_cascade.detectMultiScale(gray)  # For Detection
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Drawing Rectangle around the Face
        print(x, y)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for(ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)  # Drawing rectangle around the eyes
            print(ex, ey)
    cv2.imshow('img', img)  # Displaying the Window
    if cv2.waitKey(30) & 0xFF == ord('q'):  # By pressing the 'q' key on keyboard the execution will be terminated
        break
cap.release()
cv2.destroyAllWindows()

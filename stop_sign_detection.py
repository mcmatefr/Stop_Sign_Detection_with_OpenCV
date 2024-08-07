import cv2

# Stop Sign Cascade Classifier xml
stop_sign = cv2.CascadeClassifier('cascade_stop_sign.xml')

# Bumper Sign Cascade Classifier xml
bumper_sign = cv2.CascadeClassifier('bumpersign_classifier_haar.xml')

# Yield Sign Cascade Classifier xml
yield_sign = cv2.CascadeClassifier('yieldsign12Stages.xml')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    stop_sign_scaled = stop_sign.detectMultiScale(gray, 1.3, 5)
    bumper_sign_scaled = bumper_sign.detectMultiScale(gray, 1.3, 5)
    yield_sign_scaled = yield_sign.detectMultiScale(gray, 1.3, 5)

    # Detect the stop sign, x,y = origin points, w = width, h = height
    for (x, y, w, h) in stop_sign_scaled:
        # Draw rectangle around the stop sign
        stop_sign_rectangle = cv2.rectangle(img, (x,y),
                                            (x+w, y+h),
                                            (0, 255, 0), 3)
        # Write "Stop sign" on the bottom of the rectangle
        stop_sign_text = cv2.putText(img=stop_sign_rectangle,
                                     text="Stop Sign",
                                     org=(x, y+h+30),
                                     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                     fontScale=1, color=(0, 0, 255),
                                     thickness=2, lineType=cv2.LINE_4)

    # Detect the bumper sign, x,y = origin points, w = width, h = height
    for (x, y, w, h) in bumper_sign_scaled:
            # Draw rectangle around the bumper sign
            bumper_sign_rectangle = cv2.rectangle(img, (x, y),
                                                (x + w, y + h),
                                                (0, 255, 0), 3)
            # Write "Bumper sign" on the bottom of the rectangle
            bumper_sign_text = cv2.putText(img=bumper_sign_rectangle,
                                         text="Bumper Sign",
                                         org=(x, y + h + 30),
                                         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                         fontScale=1, color=(0, 0, 255),
                                         thickness=2, lineType=cv2.LINE_4)

    # Detect the yield sign, x,y = origin points, w = width, h = height
    for (x, y, w, h) in yield_sign_scaled:
        # Draw rectangle around the yield sign
        yield_sign_rectangle = cv2.rectangle(img, (x, y),
                                              (x + w, y + h),
                                              (0, 255, 0), 3)
        # Write "Yield sign" on the bottom of the rectangle
        yield_sign_text = cv2.putText(img=yield_sign_rectangle,
                                       text="Yield Sign",
                                       org=(x, y + h + 30),
                                       fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                       fontScale=1, color=(0, 0, 255),
                                       thickness=2, lineType=cv2.LINE_4)


    cv2.imshow("img", img)
    key = cv2.waitKey(30)
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

import cv2

harcascade="model/haarcascade_russian_plate_number.xml"

cap=cv2.VideoCapture(0) #it will capture my video and 0 is for default camera
cap.set(3,640) #width
cap.set(4,480) #height

min_area=500
count=0

while True:
    success, img=cap.read()   #to store the result and img returns the image frame captured
    plate_cascade=cv2.CascadeClassifier(harcascade)  #to load the harcascade model
    img_gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   #the model accepts only grey scale images 8gb

    plates=plate_cascade.detectMultiScale(img_gray,1.1,4) #we are giving the coordinates as parameters

    for(x,y,w,h) in plates:   #try to get area of number plate (if trying indian vala)
        area=w*h   #w=width and h=height

        if area>min_area: #considering it is a car number plate not scooty vala
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)  #we will form a rectangle with coordinates (x,y) and maximum coordinates (x+w,y+h) and the color code is of (rbg)green
    #after color code we give the thickness
            cv2.putText(img,"Number Plate",(x,y-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,255),2)

            img_roi = img[y: y + h, x:x + w]  #roi is region of interest ,it will automatically crop this portion from number plate
            cv2.imshow("ROI", img_roi)  #it will crop the area and only show cropped number plate area on the side


    cv2.imshow("Result",img)  #for displaying resultant image

    if cv2.waitKey(1) & 0xFF == ord('s'):   #when you press s key the image will be saved
        cv2.imwrite("plates/scaned_img_" + str(count) + ".jpg", img_roi)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Results", img)
        cv2.waitKey(500)
        count += 1
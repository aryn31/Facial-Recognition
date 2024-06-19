import csv
import face_recognition
import cv2
import numpy as np
from datetime import datetime
from datetime import date

video_capture=cv2.VideoCapture(1)

#load images
my_image=face_recognition.load_image_file("/Users/aryansheel/Desktop/python/facial_rec/myface.jpeg")
my_encoding=face_recognition.face_encodings(my_image)[0]

image2=face_recognition.load_image_file("/Users/aryansheel/Desktop/python/facial_rec/face_2.jpeg")
encoding2=face_recognition.face_encodings(image2)[0]
image3=face_recognition.load_image_file("/Users/aryansheel/Desktop/python/facial_rec/devesh.jpeg")
encoding3=face_recognition.face_encodings(image3)[0]

image4=face_recognition.load_image_file("/Users/aryansheel/Desktop/python/facial_rec/harsh.jpeg")
encoding4=face_recognition.face_encodings(image4)[0]

known_face_encodings=[my_encoding,encoding2,encoding3,encoding4]
known_face_names=["aryan","muthres","devesh","harsh"]

#students
students= known_face_names.copy() #pehle encoding tha to galat ara tha

#get the current date and time
today=date.today()
now=datetime.now()
current_date=datetime.now().strftime("%Y-%m-%D")
current_day = datetime.now().strftime("%A")
#strftime formats the time

f=open("/Users/aryansheel/Desktop/python/facial_rec/current_date.csv","a",newline="")
lnwriter=csv.writer(f)
if f.tell() == 0:  # Check if the file is empty
    lnwriter.writerow(["Name", "Time","Day","Date"])  # Write the header row
while True:
    _, frame=video_capture.read()
    frame=cv2.flip(frame,1)
    small_frame=cv2.resize(frame, (0,0),fx=0.25,fy=0.25)
    rgb_small_frame=cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)


    # recognize_faces
    face_location=face_recognition.face_locations(rgb_small_frame)
    face_encoding=face_recognition.face_encodings(rgb_small_frame,face_location)

    for face_encoding in face_encoding:
        matches=face_recognition.compare_faces(known_face_encodings,face_encoding)
        face_distance=face_recognition.face_distance(known_face_encodings,face_encoding)
        best_match_index=np.argmin(face_distance)

        if(matches[best_match_index]):
            name=known_face_names[best_match_index]
        #add the text if a person is present
        if name in known_face_names:
            font=cv2.FONT_HERSHEY_SIMPLEX
            bottomleftcorner = (10,100)
            fontscale=1.5
            fontcolor=(255,0,0)
            thickness=3
            linetype=2
            cv2.putText(frame, name+" Present",bottomleftcorner,font,fontscale,fontcolor,thickness,linetype)

            if name in students:
                students.remove(name)
                current_time=now.strftime("%H:%M:%S")
                lnwriter.writerow([name,current_time,current_day,today])

    cv2.imshow("attendance ",frame)
    if cv2.waitKey(1) & 0xFF ==ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()

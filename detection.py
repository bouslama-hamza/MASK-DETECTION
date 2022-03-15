from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from train_model import train_all
import datetime
import cv2

# we start by training our model
# train_all()

# load our h5 model
mymodel = load_model('trained model/mymodel.h5')

# launch our camera for detection
cap =   cv2.VideoCapture(0)

# detection miltiple faces
face_cascade    =   cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')

while cap.isOpened():

    # capture faces
    is_true , img   =   cap.read()
    face    =   face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=4)

    # transfer and reshape our detected pictures
    for(x,y,w,h) in face:
        face_img = img[y:y+h, x:x+w]
        cv2.imwrite('detected image/temp.jpg',face_img)
        test_image = image.load_img('detected image/temp.jpg',target_size=(150,150,3))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image,axis=0)

        # predict our detected image
        pred = mymodel.predict(test_image)[0][0]

        # 1 mean No Mask , 0 mean Mask
        if pred == 1:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.putText(img,'NO MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(img,'MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        datet = str(datetime.datetime.now())
        cv2.putText(img,datet,(400,450),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

    # To close our page      
    cv2.imshow('Detection',img)
    if cv2.waitKey(1) == ord('d'):
        break
    
cap.release()
cv2.destroyAllWindows()
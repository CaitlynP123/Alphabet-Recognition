import cv2 
import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 

from PIL import Image 
import PIL.ImageOps 
import os, ssl, time

x = np.load('./image.npz')['arr_0']
y = pd.read_csv("./labels.csv")["labels"]

print(pd.Series(y).value_counts()) 

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)

xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=3500, test_size=500, random_state=9) 

xTrain_scaled = xTrain/255.0 
xTest_scaled = xTest/255.0

clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(xTrain_scaled, yTrain)

yPred = clf.predict(xTest_scaled) 
accuracy = accuracy_score(yTest, yPred) 
print("The accuracy is :- ",accuracy)

cap = cv2.VideoCapture(0)

while(True): 
    try: 
        ret, frame = cap.read() 
  
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    
        height, width = gray.shape 
        upperLeft = (int(width / 2 - 56), int(height / 2 - 56)) 
        bottomRight = (int(width / 2 + 56), int(height / 2 + 56)) 
        cv2.rectangle(gray, upperLeft, bottomRight, (0, 255, 0), 2)

        roi = gray[upperLeft[1]:bottomRight[1], upperLeft[0]:bottomRight[0]] 
        
        im_pil = Image.fromarray(roi)

        image_bw = im_pil.convert('L') 
        image_bw_resized = image_bw.resize((22,30), Image.ANTIALIAS) 
        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized) 

        pixelFilter = 20

        minPixel = np.percentile(image_bw_resized_inverted, pixelFilter) 
        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-minPixel, 0, 255) 

        maxPixel = np.max(image_bw_resized_inverted) 
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/maxPixel 

        testSample = np.array(image_bw_resized_inverted_scaled).reshape(1,660) 
        testPred = clf.predict(testSample) 

        print("Predicted class is: ", testPred) 
        
        cv2.imshow('frame', gray)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break 
        
    except Exception as e: 
        pass

cap.release() 
cv2.destroyAllWindows()

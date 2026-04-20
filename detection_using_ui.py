import tkinter as tk
from tkinter.filedialog import askopenfilename
import shutil
import os
import sys
from PIL import Image, ImageTk
import tkinter.messagebox

window = tk.Tk()

window.title("Brain Hemorrhage Detection")

window.geometry("350x350")
window.configure(background ="lightblue")

title = tk.Label(text = "Click below to choose the image..", background = "lightblue", fg="Red", font=("Lucida Grande", 15))
title.grid()

    
def analysis():


    from keras.models import Sequential
    from keras.layers import Conv2D
    from keras.layers import MaxPooling2D
    from keras.layers import Flatten
    from keras.layers import Dense
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import numpy as np
    from keras.preprocessing import image
    import os
    from matplotlib import pyplot as plt
    import cv2
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from numpy import loadtxt
    from keras.models import load_model
    from sklearn.metrics import classification_report
    from keras import models
    import tensorflow as tf
    from tensorflow.keras.utils import img_to_array 
    from PIL import Image,ImageTk


    classifier = load_model('model.h5')
    classifier.summary()

    img=cv2.imread(r'img//2.jpeg',1)
    img=cv2.resize(img,(224,224))
    img=np.reshape(img,[1,224,224,3])
    print(img.shape)
    result =np.argmax(classifier.predict(img),axis=-1)
    #classifier.predict(img)
    output=cv2.imread(r'img//2.jpeg',1)
    
    #print(result.shape)
    print(result)

    if result == 0:
            tkinter.messagebox.showinfo("information","any")
            pred = 'any'
    elif result == 1:
            tkinter.messagebox.showinfo("information","epidural")
            pred = 'subarchnoid'
    elif result == 2:
            tkinter.messagebox.showinfo("information","intraventricular")
            pred = 'intraventricular'
    elif result == 3:
            tkinter.messagebox.showinfo("information","intraparenchymal")
            pred = 'intraparenchymal'
    elif result == 4:
            tkinter.messagebox.showinfo("information","subarachnoid")
            pred = 'subarachnoid'
    elif result == 5:
            tkinter.messagebox.showinfo("information","subdural")
            pred = 'subdural'


    
        
    #prediction=prediction+' : '+str(result)

    cv2.putText(output, pred, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255,0, 0), 2)
    cv2.imshow("output",output)
    cv2.waitKey(0)


def openphoto():
    dirPath = "img"
    fileList = os.listdir(dirPath)
    for fileName in fileList:  
        os.remove(dirPath + "/" + fileName)
     # C:/Users/Downloads/images is the location of the image which you want to test..... you can change it according to the image location you have  
    fileName = askopenfilename(initialdir='test dataset', title='Select image for analysis ',filetypes=[('All files', '.jpg')])
    dst = "img\\2.jpeg"
    shutil.copy(fileName, dst)
    title.destroy()
    button1.destroy()

    button2 = tk.Button(window,text="Analyse Image",bg='#ff0000', fg='#ffffff',command = analysis,height=5,width=25,font=('algerian',10,'bold'))
    button2.grid(column=0, row=500, padx=40, pady = 10)

button1 = tk.Button(text="Get image", command = openphoto,height=5,width=25,bg='#ff0000', fg='#ffffff',font=('algerian',10,'bold'))

button1.grid(column=0, row=500, padx=40, pady = 10)

window.mainloop()






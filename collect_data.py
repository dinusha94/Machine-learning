#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 12:10:28 2017

@author: dinusha
"""

#### There is a problem when using with linux not the code but the camera  #########
import cv2
import numpy as np
import Tkinter as tk       
import RPi.GPIO as GPIO
import time

ENA= 26
ENB = 27
IN1 = 19
IN2= 13
IN3 = 6
IN4 = 5
servo = 17

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(ENB, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)

p = GPIO.PWM(servo, 50)
q = GPIO.PWM(ENA, 50)
q1 = GPIO.PWM(ENB, 50)
q.start(0)
q1.start(0)
p.start(7.5)
pulse = 40

def stop():
    p.ChangeDutyCycle(7.5)
    q.ChangeDutyCycle(0)
    q1.ChangeDutyCycle(0)
    GPIO.output(IN1,GPIO.LOW)
    GPIO.output(IN2,GPIO.LOW)
    GPIO.output(IN3,GPIO.HIGH)
    GPIO.output(IN4,GPIO.HIGH)

def left():
    p.ChangeDutyCycle(6)
    q.ChangeDutyCycle(pulse)
    q1.ChangeDutyCycle(pulse)
    GPIO.output(IN1,GPIO.HIGH)
    GPIO.output(IN2,GPIO.LOW)
    GPIO.output(IN3,GPIO.LOW)
    GPIO.output(IN4,GPIO.HIGH)
    time.sleep(0.6)
    stop()

def right():
    p.ChangeDutyCycle(9)
    q.ChangeDutyCycle(pulse)
    q1.ChangeDutyCycle(pulse)
    GPIO.output(IN1,GPIO.HIGH)
    GPIO.output(IN2,GPIO.LOW)
    GPIO.output(IN3,GPIO.LOW)
    GPIO.output(IN4,GPIO.HIGH)
    time.sleep(0.6)
    stop()

def forward():
    p.ChangeDutyCycle(7.5)
    q.ChangeDutyCycle(pulse)
    q1.ChangeDutyCycle(pulse)
    GPIO.output(IN1,GPIO.HIGH)
    GPIO.output(IN2,GPIO.LOW)
    GPIO.output(IN3,GPIO.LOW)
    GPIO.output(IN4,GPIO.HIGH)
    time.sleep(0.3)
    stop()
    
class collect(tk.Frame):
        def __init__(self,master):#self is the class object variable master is the tkinter variable
                tk.Frame.__init__(self)#initiate main WINDOW for GUI
                self.master = master
        
        
                self.lbl = np.zeros((4, 4), 'float')
                for i in range(4):
                        self.lbl[i, i] = 1
        
               
                
                self.image_array = np.zeros((1, 38400))
                self.label_array = np.zeros((1, 4), 'float')
                
                self.camera = cv2.VideoCapture(0)
                self.camera.set(3,320)
                self.camera.set(4,240)
                
                self.frame = 0
                
                self.done = tk.Button(self,text = "DONE",command=self.done)
                self.done.pack(side=tk.BOTTOM)
                
                self.class1 = tk.Button(self,text = "foward",command=self.class1)
                self.class1.pack(side=tk.BOTTOM)
                
                self.class2 = tk.Button(self,text = "left",command=self.class2)
                self.class2.pack(side=tk.BOTTOM)
                
                self.class3 = tk.Button(self,text = "right",command=self.class3)
                self.class3.pack(side=tk.BOTTOM)
                
                self.class4 = tk.Button(self,text = "stop",command=self.class4)
                self.class4.pack(side=tk.BOTTOM)
                
                
        
                        
        
        def class1(self):
                self.collect_image()
                self.image_array = np.vstack((self.image_array, self.temp_array))
                self.label_array = np.vstack((self.label_array, self.lbl[2]))
                forward()
                print(self.lbl[2])
                
        
        def class2(self):
                self.collect_image()
                self.image_array = np.vstack((self.image_array, self.temp_array))
                self.label_array = np.vstack((self.label_array, self.lbl[3]))
                left()
                print(self.lbl[0])
                
        
        def class3(self):
                self.collect_image()
                self.image_array = np.vstack((self.image_array, self.temp_array))
                self.label_array = np.vstack((self.label_array, self.lbl[0]))
                right()
                print(self.lbl[1])
                
        
        def class4(self):
                self.collect_image()
                self.image_array = np.vstack((self.image_array, self.temp_array))
                self.label_array = np.vstack((self.label_array, self.lbl[1]))
                stop()
                print(self.lbl[3])
                
        
        def done(self):
                self.train = self.image_array[1:, :]
                self.train_labels = self.label_array[1:, :]
                # save training data as a numpy file
                np.savez('data_npz/train.npz', train=self.train, train_labels=self.train_labels)
                print("number of frames: "+str(self.frame))
                self.master.destroy()  

        def collect_image(self):
                 
                _, self.image = self.camera.read()
                self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                # select lower half of the image
                self.roi = self.gray[120:320, :]
                # save streamed images
                cv2.imwrite("data_images/"+str(self.frame)+".jpg", self.image)
##                cv2.imshow('full window',self.image)
##                cv2.imshow('roi_window',self.roi)
                # reshape the roi image into one row array
                self.temp_array = self.roi.reshape(1, 38400).astype(np.float32)
        
                self.frame += 1
                







if __name__ == "__main__":
       
        root = tk.Tk()
        collect(root).pack()
        root.mainloop()


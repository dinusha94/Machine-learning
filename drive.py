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
    q.ChangeDutyCycle(22.5)
    q1.ChangeDutyCycle(22.5)
    GPIO.output(IN1,GPIO.HIGH)
    GPIO.output(IN2,GPIO.LOW)
    GPIO.output(IN3,GPIO.LOW)
    GPIO.output(IN4,GPIO.HIGH)

def right():
    p.ChangeDutyCycle(9)
    q.ChangeDutyCycle(22.5)
    q1.ChangeDutyCycle(22.5)
    GPIO.output(IN1,GPIO.HIGH)
    GPIO.output(IN2,GPIO.LOW)
    GPIO.output(IN3,GPIO.LOW)
    GPIO.output(IN4,GPIO.HIGH)

def forward():
    p.ChangeDutyCycle(7.5)
    q.ChangeDutyCycle(22.5)
    q1.ChangeDutyCycle(22.5)
    GPIO.output(IN1,GPIO.HIGH)
    GPIO.output(IN2,GPIO.LOW)
    GPIO.output(IN3,GPIO.LOW)
    GPIO.output(IN4,GPIO.HIGH)

cap = cv2.VideoCapture(0)
cap.set(3,320)
cap.set(4,240)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    print cap.get(cv2.cv.CV_CAP_PROP_FPS)
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


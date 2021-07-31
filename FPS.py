# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 15:29:32 2017

@author: geaxx
"""
import time
import cv2

def now():
    return time.perf_counter()
    
class FPS: # To measure the number of frame per second
    def __init__(self, mean_nb_frames=10):
        self.nbf = -1
        self.fps = 0
        self.start = 0
        self.stop = 0
        self.local_start = 0
        self.mean_nb_frames = mean_nb_frames
        
    def update(self):
        if self.nbf%self.mean_nb_frames == 0:
            if self.start != 0:
                self.stop = now()
                self.fps = self.mean_nb_frames/(self.stop-self.local_start)
                self.local_start = self.stop
            else :
                self.start = self.local_start = now()    
        self.nbf+=1
    
    def get(self):
        return self.fps
    
    def get_global(self):
        if self.stop == 0: self.stop = now()
        return self.nbf/(self.stop-self.start)

    def draw(self, win, orig=(10,30), font=cv2.FONT_HERSHEY_SIMPLEX, size=2, color=(0,255,0), thickness=2):
        cv2.putText(win,f"FPS={self.get():.2f}",orig,font,size,color,thickness)


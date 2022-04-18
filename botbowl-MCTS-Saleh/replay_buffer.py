import numpy as np
from scipy.ndimage.interpolation import shift
import random

class Buffer():
    def __init__(self,max_len):
        self.memory = []# spacials, non_spacials, act_dis, win_rate
        self.max_len = max_len


    def add_to_buffer(self,newMemory):

        spacials, non_spacials, act_dis, win_rate  = newMemory[0],newMemory[1],newMemory[2],newMemory[3]
        if len(self.memory) > self.max_len:
            keep_this = self.max_len - spacials.shape[0]
            self.memory = self.memory[-keep_this:]

        # self.memory = np.concatenate((self.memory,newMemory),axis=0)
        for i in range(spacials.shape[0]):
            self.memory.append((spacials[i],non_spacials[i],act_dis[i],win_rate[i]))
        
        print('size of buffer: ',len(self.memory))


        # print(sorted(act_dis[0], reverse=True)[:10])


    def get_from_buffer(self,len):
        batch = random.choices(self.memory,k=len)
        spacials = np.array([i[0] for i in batch])
        non_spacials = np.array([i[1] for i in batch])
        act_dis  = np.array([i[2] for i in batch])
        win_rate = np.array([i[3] for i in batch])

        print('got form buffer spacials shape: ',spacials.shape)
        print('got form buffer non_spacials shape: ',non_spacials.shape)
        return spacials,non_spacials,act_dis,win_rate

    def set_len(self, new_len):
        self.max_len = new_len
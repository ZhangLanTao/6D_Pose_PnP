import numpy as np
from average_quaternions import averageQuaternions
from scipy.spatial.transform import Rotation

class slide_window_filter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.need_init = True
        self.rotation_history = np.zeros((window_size, 4))
        self.translation_history = np.zeros((window_size, 3))
        self.fake_pointer = 0


    def update(self, new_rotation, new_translation):
        self.rotation_history[self.fake_pointer] = new_rotation.as_quat()
        self.translation_history[self.fake_pointer] = new_translation
        self.fake_pointer = (self.fake_pointer+1)//self.window_size
        if self.need_init:
            if self.fake_pointer == 0:      # 已经写完一遍，指针回到0后
                self.need_init = False
            return new_rotation, new_translation
        else:
            return averageQuaternions(self.rotation_history), np.average(self.translation_history, axis=0)


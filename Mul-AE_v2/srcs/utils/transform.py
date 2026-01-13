from PIL import Image
import torch

class Transform(object):
    def __init__(self, random_horizon_flip=0.0, rotation=None, resize=None):
        super().__init__()
        self.random_horizon_flip = random_horizon_flip
        self.rotation = rotation
        self.resize = resize

        self.rot = 0.0

    def __call__(self, img):        
        if torch.rand(1) < self.random_horizon_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        if not self.rot == 0.0:
            x, y, w, h = [0, 0, img.size[0], img.size[1]]
            center_x = x+w/2.0
            center_y = y+h/2.0
            img = img.rotate(self.rot, center=(center_x, center_y))
        
        if self.resize:
            img = img.resize(self.resize)

        return img

    def random_all_factiors(self):
        if self.rotation:
            self.rot = torch.empty(1).uniform_(self.rotation[0], self.rotation[1])
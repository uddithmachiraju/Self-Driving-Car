import pygame 
import numpy as np 

# ---------- Functions --------- # 

def scale_image(image, scale_factor):
    size = round(image.get_width() * scale_factor), round(image.get_height() * scale_factor)
    return pygame.transform.scale(image, size)

def rotate_vector_ACW(vector, angle):
    angle = angle * np.pi / 180
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)] 
        ]
    )

    return rotation.dot(vector)

# ---------- WINDOW SETTINGS --------- # 
WINDOW_WIDTH = 500 
WINDOW_HEIGHT = 500 

HALF_WIDTH = WINDOW_WIDTH // 2 
HALF_HEIGHT = WINDOW_HEIGHT // 2

# ---------- CAMARA AND CONTROLS ---------- # 
FPS = 30
CAMARA_BORDERS = {
    'LEFT' : 250,
    'RIGHT' : 250,
    'TOP' : 250,
    'BOTTOM' : 250
}

# ---------- ASSETS ---------- # 
CAR = scale_image(pygame.image.load('../Assets/green-car.png'), 0.8)
BACKGROUND = scale_image(pygame.image.load('../Assets/Road.png'), 0.6) 
BACKGROUND_MASK = scale_image(pygame.image.load('../Assets/Road_Mask.png'), 0.6) 
GRASS = pygame.image.load('../Assets/Grass.png') 
import cv2
import pygame, sys 
import numpy as np 
from settings import * 

pygame.init()

class Environment:
    def __init__(self, controller):  
        self.controller = controller
        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption('Car Racing')
        self.clock = pygame.time.Clock()

        # Camera rectangle with borders
        self.camera_rect = pygame.Rect(CAMARA_BORDERS['LEFT'], 
                                       CAMARA_BORDERS['TOP'], 
                                       WINDOW_WIDTH - CAMARA_BORDERS['LEFT'] - CAMARA_BORDERS['RIGHT'], 
                                       WINDOW_HEIGHT - CAMARA_BORDERS['TOP'] - CAMARA_BORDERS['BOTTOM'])

        # Offset vector to control background movement
        self.offset = pygame.math.Vector2(0, 0)
        self.track_mask = pygame.surfarray.array2d(BACKGROUND_MASK)

        # Car and background
        self.car_image = CAR
        self.background_size = BACKGROUND.get_size()
        self.background_position = pygame.math.Vector2(0, 0) 
    
    def display_state(self, state_image):
        display_image = (state_image * 255).astype(np.uint8)
        cv2.imshow("State Image", display_image)
        cv2.waitKey(1)

    def box_target_camera(self, target):
        if target.position[0] < self.camera_rect.left:
            self.camera_rect.left = target.position[0]
        if target.position[0] > self.camera_rect.right:
            self.camera_rect.right = target.position[0]
        if target.position[1] < self.camera_rect.top:
            self.camera_rect.top = target.position[1]
        if target.position[1] > self.camera_rect.bottom:
            self.camera_rect.bottom = target.position[1]

        # Calculate offset based on the camera rectangle
        self.offset.x = self.camera_rect.left - CAMARA_BORDERS['LEFT'] 
        self.offset.y = self.camera_rect.top - CAMARA_BORDERS['TOP']

    def draw(self):
        self.box_target_camera(self.controller)
        for x in range(-self.background_size[0], WINDOW_WIDTH, self.background_size[0]):
            for y in range(-self.background_size[1], WINDOW_HEIGHT, self.background_size[1]):
                background_pos = pygame.math.Vector2(x, y) - self.offset
                self.window.blit(GRASS, background_pos)

        background_pos = self.background_position - self.offset
        self.window.blit(BACKGROUND, background_pos)

        rotated_car = pygame.transform.rotate(self.car_image, self.controller.rotation)
        car_rect = rotated_car.get_rect(center=(self.controller.position[0] - self.offset.x, 
                                                self.controller.position[1] - self.offset.y))
        self.window.blit(rotated_car, car_rect) 

    def run(self):
        while True:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.controller.update_position() 
            self.draw()
            pygame.display.update() 

class Controller:
    def __init__(self, position):
        self.steering_angle = 0
        self.position = position
        self.rotation = 0 
        self.velocity = 0
        self.acceleration = 0
        self.total_reward = 0

    def rotate_and_move(self):
        radius = 1.5 / np.sin(self.steering_angle * np.pi / 180) if self.steering_angle else 1e8
        for _ in range(3):
            if self.velocity:
                self.rotation += 2 / radius
            self.position += rotate_vector_ACW(np.array([0, self.velocity]), -self.rotation)
        self.velocity += self.acceleration 

    def calculate_reward(self):
        reward = 0
        on_track = self.is_on_track() 
        reward += 0.2 * self.velocity if on_track else -20
        if self.velocity > 0:
            reward -= 0.1 * abs(self.velocity)
        self.total_reward += reward
        return reward

    def is_on_track(self):
        x, y = int(self.position[0]), int(self.position[1])
        if 0 <= x < BACKGROUND_MASK.get_width() and 0 <= y < BACKGROUND_MASK.get_height():
            road_mask = pygame.mask.from_surface(BACKGROUND_MASK)
            return road_mask.get_at((x, y)) != 1
        return True

    def human_controls(self):
        keys = pygame.key.get_pressed()
        self.acceleration = -0.02 if keys[pygame.K_UP] else (0.02 if keys[pygame.K_DOWN] else 0)
        self.steering_angle = -30 if keys[pygame.K_RIGHT] else (30 if keys[pygame.K_LEFT] else 0)

    def ai_controls(self, action):
        self.acceleration = -0.02 if action == 0 else (0.02 if action == 1 else 0)
        self.steering_angle = -30 if action == 2 else (30 if action == 3 else 0)

    def get_state(self, window):
        screen_array = pygame.surfarray.array3d(window)
        screen_array = np.transpose(screen_array, (1, 0, 2))
        gray_image = cv2.cvtColor(screen_array, cv2.COLOR_RGB2GRAY)
        resized_image = cv2.resize(gray_image, [128, 128], interpolation=cv2.INTER_AREA) 
        normalized_image = np.array(resized_image / 255.0, dtype=np.float32) 
        return normalized_image
    
    def step_srt(self, action, window):  
        self.update_position(controls='ai', action=action) 
        state = self.get_state(window)
        reward = self.calculate_reward()
        terminated = self.is_on_track() 
        return state, reward, terminated

    def update_position(self, controls='human', action=None):
        if controls == 'human':
            self.human_controls()
        else:
            self.ai_controls(action)
        self.rotate_and_move()

class CarRacing(Environment):
    def __init__(self, controller):
        super().__init__(controller) 

    def reset(self):
        self.controller.steering_angle = 0
        self.controller.position = (155, 400)
        self.controller.rotation = 0 
        self.controller.velocity = 0
        self.controller.acceleration = 0
        self.controller.total_reward = 0
        
    def step(self, action): 
        if action == 'Initial':
            state = self.controller.get_state(self.window)
            reward = 0
            terminated = False
        else:
            state, reward, terminated = self.controller.step_srt(action, self.window) 

        self.clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.controller.update_position() 
        self.draw()
        pygame.display.update()
        
        return state, reward, terminated
    
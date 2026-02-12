import pygame
import os

class AlertSystem:
    def __init__(self, sound_path):
        self.enabled = False
        if os.path.exists(sound_path):
            pygame.mixer.init()
            self.sound = pygame.mixer.Sound(sound_path)
            self.enabled = True
        else:
            print("⚠️ Alarm sound not found, alerts will be silent")

    def play(self):
        if self.enabled and not pygame.mixer.get_busy():
            self.sound.play()


# def play_alert():
#     print("ALERT: Driver appears drowsy!")

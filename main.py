import cv2 as cv
import pyautogui
import time
import numpy as np
import keyboard
from vision import Vision
cascade_npc = cv.CascadeClassifier('cascade/cascade.xml')
vision_npc = Vision(None)

while True:
    screenshot = pyautogui.screenshot()
    screenshot = cv.cvtColor(
        src=np.array(screenshot), code=cv.COLOR_RGB2BGR
    )
    loop_time = time.time()

    # Verifica se a captura de tela foi bem-sucedida
    valores = cascade_npc.detectMultiScale(
        screenshot, scaleFactor=1.5, minNeighbors=4)
    detection_image = vision_npc.draw_rectangles(screenshot, valores)
    # Converte a captura de tela em uma matriz NumPy

    cv.imwrite("log/{}.jpg".format(loop_time), detection_image)

    if keyboard.is_pressed('q'):
        cv.destroyAllWindows()
        break
    elif keyboard.is_pressed('f'):
        print("f key pressed")
        cv.imwrite("positive/{}.jpg".format(loop_time), screenshot)
    elif keyboard.is_pressed('d'):
        print("d key pressed")
        cv.imwrite("negative/{}.jpg".format(loop_time), screenshot)

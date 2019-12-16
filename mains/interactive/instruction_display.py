import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw
import textwrap

from visualization import Presenter

H = 1080
W = 1920

GAP = 100

NOOP = True


class InstructionDisplay:
    def __init__(self):
        self.presenter = Presenter()
        self.background = np.zeros((1080, 1920, 3))
        # BGR format
        self.background[:, :, 0], self.background[:, :, 1], self.background[:, :, 2] = 0xd7, 0x92, 0x28
        self.background = self.background.astype(np.uint8)
        self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-ExtraLight.ttf", 60)

        if not NOOP:
            cv2.namedWindow('Instruction', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Instruction', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def show_instruction(self, instruction_str):
        bg_pil = Image.fromarray(self.background.copy())
        draw = ImageDraw.Draw(bg_pil)
        instruction_str = instruction_str[0].upper() + instruction_str[1:]
        instruction_str.replace("  ", " ")

        lines = textwrap.wrap(instruction_str, width=60)
        dims = [draw.textsize(line, self.font) for line in lines]
        total_height = sum([d[1] for d in dims])
        print(lines)

        #w, h = draw.textsize(instruction_str, font=self.font)

        stack_h = 0
        for line, dim in zip(lines, dims):
            draw.text((int((W - dim[0]) / 2), int((H - total_height + stack_h) / 2)), line, (255, 255, 255), font=self.font)
            stack_h += dim[1] + GAP

        display = np.array(bg_pil)

        if not NOOP:
            cv2.imshow("Instruction", display)
            cv2.waitKey(1)

    def tick(self):
        # This has to be called repeatedly to run the CV2 event loop and prevent the display from locking up
        if not NOOP:
            cv2.waitKey(1)

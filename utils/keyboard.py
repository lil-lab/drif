from pykeyboard import PyKeyboardEvent
import threading

KEYFUNC = {
    "a": "left",
    "s": "backward",
    "d": "right",
    "w": "forward",
    " ": "stop"
}

FWD_VEL = 1
TURN_VEL = 1.0

class MonitorSuper(PyKeyboardEvent):

    def __init__(self):
        PyKeyboardEvent.__init__(self)
        self.forward = False
        self.backward = False
        self.left = False
        self.right = False
        self.current_action = [0, 0, 0, 0]

    def reset(self):
        self.current_action = [0, 0, 0, 0]

    def tap(self, keycode, c, press):
        '''Monitor Super key.'''
        if c in KEYFUNC:
            #print (KEYFUNC[c])
            if KEYFUNC[c] == "left":
                if (press):
                    self.current_action[2] = -TURN_VEL
                else:
                    self.current_action[2] = 0
            if KEYFUNC[c] == "right":
                if press:
                    self.current_action[2] = TURN_VEL
                else:
                    self.current_action[2] = 0
            if KEYFUNC[c] == "forward":
                if press:
                    self.current_action[0] = FWD_VEL
                else:
                    self.current_action[0] = 0
            if KEYFUNC[c] == "backward":
                if press:
                    self.current_action[0] = -FWD_VEL
                else:
                    self.current_action[0] = 0

            if KEYFUNC[c] == "stop":
                if press:
                    self.current_action[3] = 1.0


class KeyTeleop():

    def __init__(self):
        self.mon = MonitorSuper()
        self.thread = threading.Thread(target=self.run, args=())
        self.thread.daemon = True
        self.thread.start()

    def run(self):
        self.mon.run()

    def get_command(self):
        return self.mon.current_action

    def reset(self):
        self.mon.reset()
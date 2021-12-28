import os
import sys
import json
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
sys.path.insert(0, '/home/drone/drone-sim/')
sys.path.insert(0, '/home/drone/catkin_ws/src/drone-lab')
from tkinter import *
from config_env.definitions.landmarks import get_landmark_names
from config_env.config_env import get_config_dir, get_all_drone_images_dir, get_drone_images_dir
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import imread
from PIL import ImageTk, Image

all_landmarks = get_landmark_names()

def get_hand_labeling_dir():
    return "/home/drone/catkin_ws/data/hand_labeling"

def get_hand_labeling_file(env_id, img_id):
    return os.path.join(get_hand_labeling_dir(), "flight_{}".format(env_id), "label_{}.json".format(img_id))



def relabel(env_ids, images, labels, img_ids):
    for i_img, env_id in enumerate(env_ids):
        variables = []
        master = Tk()

        master.title("Start tagging")
        landmarks = []
        positions = []

        def showxy(event):
            '''
            show x, y coordinates of mouse click position
            event.x, event.y relative to ulc of widget (here root)
            '''
            # xy relative to ulc of root
            # xy = 'root x=%s  y=%s' % (event.x, event.y)
            # optional xy relative to blue rectangle

            xy = 'x={}  y={}'.format(event.x, event.y)
            positions.append([event.x, event.y])
            master.title(xy)


        img_path = os.path.join(get_drone_images_dir(env_ids[i_img]), 'usb_cam_{}.jpg'.format(str(img_ids[i_img])))
        w = 800
        h = 400
        cv = Canvas(master, width=w, height=h, bg='white')
        cv.grid(row=0, sticky=W)

        img = ImageTk.PhotoImage(Image.open(img_path))
        cv.create_image(0, 0, image=img, anchor="nw")
        cv.bind( '<Button-1>', showxy)

        #print(img_path)
        #panel = Label(master, image = img)
        #panel.grid(row=0, sticky=W)

        for i, lm in enumerate(all_landmarks):
            if not(labels[i_img] is None):
                if lm in labels[i_img]:
                    var=IntVar()
                    var.set(1)
                else:
                    var = IntVar()
            else:
                var = IntVar()
            variables.append(var)
            Checkbutton(master, text=lm, variable=var).grid(row=i+1, sticky=W)


        def save_labeling():
            flight_dir = os.path.join(get_hand_labeling_dir(), "flight_{}".format(env_id))
            if not(os.path.exists(flight_dir)):
                os.mkdir(flight_dir)
            f = get_hand_labeling_file(env_id, img_ids[i_img])
            mask = np.where([v.get() for v in variables])[0]
            landmarks = [all_landmarks[m] for m in mask]
            if len(landmarks)==len(positions):
                lm_on_img = {'landmarks': landmarks, 'lm_pos_fpv': positions}
            else:

            print(lm_on_img)
            with open(f, 'w') as fo:
                json.dump(lm_on_img, fo)

        def restart_tagging():
            positions = []
            master.title("Start tagging")

        Button(master, text='Save', command=save_labeling).grid(row=i+2, sticky=W)
        Button(master, text='Restart location tagging', command=restart_tagging).grid(row=i + 4, sticky=W)
        Button(master, text='Quit', command=master.destroy).grid(row=i+3, sticky=W)
        master.mainloop()

if __name__=='__main__':
    env_ids = [89, 89]
    img_ids = np.arange(2, 80, 5)
    labels = [['Mushroom'],None]
    images = [imread(os.path.join(get_drone_images_dir(env_ids[i]), 'usb_cam_{}.jpg'.format(str(img_ids[i]) ))) for i in range(len(env_ids))]
    relabel(env_ids, images, labels, img_ids)

from tkinter import *
from tkinterdnd2 import *
from PIL import ImageTk,Image
import numpy as np
import cv2
import io
import os
import matplotlib.pyplot as plt

# Se citesc toate etichetele ce se afla in fisierul coco.names (tot ce se poate detecta)
labels = open('coco.names').read().strip().split('\n')
print("These are all the possible objects/animals that can be detected:")
print(labels)

#incarcam greutatiile si configuratiile pentru yolo
# YOLO (You Only Look Once)
weights_path = os.getcwd() + '\yolov3.weights'
configuration_path = os.getcwd() + "\yolov3.cfg"

# variabile predefinite
probability_minimum = 0.5
threshold = 0.3

# connect to the neural network
# Darknet: Open Source Neural Networks in C
network = cv2.dnn.readNetFromDarknet(configuration_path, weights_path)
layers_names_all = network.getLayerNames()
layers_names_output = [layers_names_all[i-1] for i in network.getUnconnectedOutLayers()]

def DropImage(event):
    testvariable.set(event.data)
    window.file_name=testvariable.get()
    # Citire imagine
    image_path = window.file_name
    image_input = cv2.imread(image_path)

    # Preprocesare imagine
    blob = cv2.dnn.blobFromImage(image_input, 1/255.0, (416,416), swapRB=True, crop=False)
    blob_to_show = blob[0,:,:,:].transpose(1,2,0)
    network.setInput(blob)
    output_from_network = network.forward(layers_names_output)
    np.random.seed(42)
    colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    bounding_boxes = []
    confidences = []
    class_numbers = []
    h,w = image_input.shape[:2]

    # Trecere prin reteaua Yolo
    for result in output_from_network:
        for detection in result:
            scores = detection[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]
            if confidence_current > probability_minimum:
                box_current = detection[0:4] * np.array([w, h, w, h])
                x_center, y_center, box_width, box_height = box_current.astype('int')
                x_min = int(x_center-(box_width/2))
                y_min = int(y_center-(box_height/2))
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

    if len(class_numbers)!=0:
        entityName=labels[class_numbers[-1]]
    else:
        entityName="Unknown"
    print("This is : "+entityName)
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)
    
    # Setare bounding box
    finalProbability=0
    if len(results) > 0:
        for i in results.flatten():
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
            colour_box_current = [int(j) for j in colours[class_numbers[i]]]
            cv2.rectangle(image_input, (x_min, y_min), (x_min + box_width, y_min + box_height),
                        colour_box_current, 2)
            finalProbability = confidences[i]
            text_box_current = '{}'.format(entityName)
            cv2.putText(image_input, text_box_current, (x_min, y_min + 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, colour_box_current, 2)

    str = "With a probability of"
    print(str, finalProbability)

    # Trimitere imagine catre TinkerDnD
    plt.axis('off')
    plt.rcParams['figure.figsize'] = (6.0,6.0)
    plt.imshow(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg')
    buf.seek(0)
    im = Image.open(buf)
    print(window.file_name)
    # resize image
    reside_image = im.resize((600, 600), resample=Image.BILINEAR)
    # displays an image
    window.image = ImageTk.PhotoImage(reside_image)
    image_label = Label(labelframe, image=window.image).place(x=20, y=10)


window = TkinterDnD.Tk()
window.title('Delftstack')
window.geometry('700x700')
window.config(bg='gold')

testvariable = StringVar()
textlabel=Label(window, text='drop the file here', bg='#fcba03')
textlabel.pack(anchor=NW, padx=10)
entrybox = Entry(window, textvar=testvariable, width=80)
entrybox.pack(fill=X, padx=10)
entrybox.drop_target_register(DND_FILES)
entrybox.dnd_bind('<<Drop>>', DropImage)

labelframe = LabelFrame(window, bg='gold')

labelframe.pack(fill=BOTH, expand=True, padx=9, pady=9)

window.mainloop()
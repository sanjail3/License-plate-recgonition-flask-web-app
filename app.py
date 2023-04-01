#License Plate Recgonition Using LPRnet


from flask import Flask, redirect, url_for, render_template, request
import os
from Licenseplate import LicensePlateDetector
import torch
from License_Plate_Recognition.model.LPRNet import build_lprnet
from License_Plate_Recognition.test_LPRNet import Greedy_Decode_inference
import cv2
import numpy as np
import matplotlib.pyplot as plt




lprnet = build_lprnet(lpr_max_len=16, class_num=37).eval()
lprnet.load_state_dict(
        torch.load("weights/best_lprnet.pth", map_location=torch.device("cpu"))
    )





lpd = LicensePlateDetector(
    pth_weights='C:/Users/sanjai/PycharmProject/license plateflaskapp/yolov3-tiny-obj_last.weights',
    pth_cfg='C:/Users/sanjai/PycharmProject/license plateflaskapp/yolov3-tiny-obj.cfg',
    pth_classes='C:/Users/sanjai/PycharmProject/license plateflaskapp/classes.txt'
)



app = Flask(__name__)

UPLOAD_PATH ='photos/'
plates = []

@app.route('/',methods=['POST','GET'])
def index():
    if request.method == 'POST':
        upload_file = request.files['image_name']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH,filename)
        upload_file.save(path_save)
        val=lpd.detect(path_save)
        if val !=1:
            # Plot original image with rectangle around the plate
            plt.figure(figsize=(24, 24))
            plt.imshow(cv2.cvtColor(lpd.fig_image, cv2.COLOR_BGR2RGB))
            plt.savefig('detected.jpg')

            lpd.crop_plate()
            plt.figure(figsize=(10, 4))
            plt.imshow(cv2.cvtColor(lpd.roi_image, cv2.COLOR_BGR2RGB))
            plt.savefig('static/detected1.jpg')


            roi_n = cv2.imread('static/detected1.jpg')
            plate_images = []
            im = cv2.resize(roi_n, (94, 24)).astype("float32")
            im -= 127.5
            im *= 0.0078125
            im = torch.from_numpy(np.transpose(im, (2, 0, 1)))
            plate_images.append(im)

            plate_labels = Greedy_Decode_inference(lprnet, torch.stack(plate_images, 0))
            text = str(plate_labels[0])
        else:
            text="No number Plate detected"
            return render_template('index1.html', upload=True,text=text)


        return render_template("index.html", upload=True, upload_image=filename, text=text)

    return render_template('index.html',upload=False)




if __name__ == '__main__':
    # DEBUG is SET to TRUE. CHANGE FOR PROD
    app.run()
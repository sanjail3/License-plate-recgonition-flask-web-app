#License Plate reccgonition using pytesseract


import imutils
from flask import Flask, redirect, url_for, render_template, request

import os

from Licenseplate import LicensePlateDetector


import pytesseract

lpd = LicensePlateDetector(
    pth_weights='C:/Users/sanjai/PycharmProject/license plateflaskapp/yolov3-tiny-obj_last.weights',
    pth_cfg='C:/Users/sanjai/PycharmProject/license plateflaskapp/yolov3-tiny-obj.cfg',
    pth_classes='C:/Users/sanjai/PycharmProject/license plateflaskapp/classes.txt'
)


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract as pt







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
            roi = cv2.imread('static/detected1.jpg')



            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

            img = cv2.imread('static/detected1.jpg', cv2.IMREAD_COLOR)
            img = cv2.resize(img, (600, 400))

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 13, 15, 15)

            edged = cv2.Canny(gray, 30, 200)
            contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            screenCnt = None

            for c in contours:

                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.018 * peri, True)

                if len(approx) == 4:
                    screenCnt = approx
                    break

            if screenCnt is None:
                detected = 0
                print("No contour detected")
            else:
                detected = 1

            if detected == 1:
                cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

            mask = np.zeros(gray.shape, np.uint8)
            new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
            new_image = cv2.bitwise_and(img, img, mask=mask)

            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))
            Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]
            cv2.imwrite("new.png",Cropped)
            text = pytesseract.image_to_string(Cropped, config='--psm 11')

            roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
            # reader = easyocr.Reader(['en'])
            # license_plate_gray = cv2.cvtColor(roi_n, cv2.COLOR_BGR2GRAY)
            #
            # _, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)
            #
            # output = reader.readtext(license_plate_thresh)
            # list_t=[]
            # final_text=''
            # for out in output:
            #     text_bbox, text, text_score = out
            #     if text_score > 0.5:
            #         print(text, text_score)
            #         if len(text)>len(final_text):
            #             final_text=text
            # print("Detected license plate Number is:", final_text)
        else:
            text="No number Plate detected"
            return render_template('index1.html', upload=True,text=text)


        return render_template("index.html", upload=True, upload_image=filename, text=text)

    return render_template('index.html',upload=False)




if __name__ == '__main__':
    # DEBUG is SET to TRUE. CHANGE FOR PROD
    app.run()
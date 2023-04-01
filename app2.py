#License Plate reccgonition using EasyOCR


from flask import Flask, redirect, url_for, render_template, request

import os

from Licenseplate import LicensePlateDetector

import easyocr




lpd = LicensePlateDetector(
    pth_weights='C:/Users/sanjai/PycharmProject/license plateflaskapp/yolov3-tiny-obj_last.weights',
    pth_cfg='C:/Users/sanjai/PycharmProject/license plateflaskapp/yolov3-tiny-obj.cfg',
    pth_classes='C:/Users/sanjai/PycharmProject/license plateflaskapp/classes.txt'
)





import cv2

import matplotlib.pyplot as plt








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




            reader = easyocr.Reader(['en'])
            license_plate_gray = cv2.cvtColor(roi_n, cv2.COLOR_BGR2GRAY)

            _, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

            output = reader.readtext(license_plate_thresh)
            list_t=[]
            final_text=''
            for out in output:
                text_bbox, text, text_score = out
                if text_score > 0.5:
                    print(text, text_score)
                    if len(text)>len(final_text):
                        final_text=text
            print("Detected license plate Number is:", final_text)
        else:
            text="No number Plate detected"
            return render_template('index1.html', upload=True,text=text)


        return render_template("index.html", upload=True, upload_image=filename, text=final_text)

    return render_template('index.html',upload=False)




if __name__ == '__main__':
    # DEBUG is SET to TRUE. CHANGE FOR PROD
    app.run()
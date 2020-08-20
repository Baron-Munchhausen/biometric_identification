import cv2, os
import numpy as np
import math
from PIL import Image

#
#   Пример, взятый за основу поиска лица
#
#   http://hanzratech.in/2015/02/03/face-recognition-using-opencv.html
#


def detect_face(path):
    
    cascade_face = cv2.CascadeClassifier("./models/haarcascade_frontalface_default.xml")
    cascade_eyes = cv2.CascadeClassifier("./models/haarcascade_eye.xml")
    cascade_nose = cv2.CascadeClassifier("./models/haarcascade_mcs_nose.xml")
    cascade_mouth = cv2.CascadeClassifier("./models/haarcascade_mcs_mouth.xml")

    image_paths = [os.path.join(path, f) for f in os.listdir(path)]

    images = []
    labels = []

    for image_path in image_paths:

        # Черно-белое изображение
        gray = Image.open(image_path).convert('L')
        image = np.array(gray, 'uint8')
        

        filename_w_ext = os.path.basename(image_path)
        filename, file_extension = os.path.splitext(filename_w_ext)

        # Лицо
        faces = cascade_face.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Поиск элементов лица
        for (x, y, w, h) in faces:

            face = image[y: y + h, x: x + w]

            eyes = cascade_eyes.detectMultiScale(face)
            count_eyes = 0
            for (ex,ey,ew,eh) in eyes:
                 cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(255,0,0),1)
                 count_eyes += 1

            if len(eyes) == 2:
                if eyes[0][0] < eyes[1][0]:
                    left_center = (eyes[0][0] + (eyes[0][2]/2), eyes[0][1] + (eyes[0][3]/2))
                    right_center = (eyes[1][0] + (eyes[1][2]/2), eyes[1][1] + (eyes[1][3]/2))
                else:
                    left_center = (eyes[1][0] + (eyes[1][2]/2), eyes[1][1] + (eyes[1][3]/2)) 
                    right_center = (eyes[0][0] + (eyes[0][2]/2), eyes[0][1] + (eyes[0][3]/2))

                edge1 = np.int0((right_center[0] - left_center[0],right_center[1] - left_center[1]))
                reference = (1,0)

                angle = 180.0/math.pi * math.acos((reference[0]*edge1[0] + reference[1]*edge1[1]) / (cv2.norm(reference) *cv2.norm(edge1)))
                angle = 0 - angle

                image = rotateImage(image, angle)
                
                faces_result = cascade_face.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces_result:

                    face_result = image[y: y + h, x: x + w]


            #noses = cascade_nose.detectMultiScale(face)
            #for (ex,ey,ew,eh) in noses:
            #     cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)

            #mouthes = cascade_mouth.detectMultiScale(face)
            #for (ex,ey,ew,eh) in mouthes:
            #     cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,0,255),1)

            images.append(face_result)
            labels.append(filename_w_ext)


            # В окне показываем изображение
            cv2.imshow("", face_result)
            cv2.imwrite(os.path.join(path, filename + '_detected.' + file_extension), face_result)
            cv2.waitKey(50)

    return images, labels

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result
        




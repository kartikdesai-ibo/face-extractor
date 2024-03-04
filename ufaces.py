import os
import cv2
import argparse
import filetype as ft
import numpy as np
from pathlib import Path
from PIL import Image
from facedetector import FaceDetector
from cv2 import dnn_superres
sr = dnn_superres.DnnSuperResImpl_create()
path = "EDSR_x4.pb"
sr.readModel(path)
import cv2
import dlib
import os
from skimage import io
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

 
def get_face_encodings(image_path):

    detector = dlib.get_frontal_face_detector()

    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    img = io.imread(image_path)

    dets = detector(img, 1)

    face_encodings = []

    for det in dets:

        shape = predictor(img, det)

        face_encoding = face_encoder.compute_face_descriptor(img, shape)

        face_encodings.append(face_encoding)

    return face_encodings
 
def compare_faces(encodings1, encodings2):

    distance_matrix = euclidean_distances(encodings1, encodings2)

    return distance_matrix

def find_unique_faces(image_paths):
    face_encodings_by_image = {}
    
    face_encodings_by_image_unique_flag = dict()
    face_encodings_by_image_test = dict()
    face_encodings_set = {}
    unique_faces = set()
    
    for image_path in image_paths:
        face_encodings = get_face_encodings(image_path)
        face_encodings_by_image[image_path] = face_encodings
        face_encodings_by_image_test[image_path] = face_encodings
        face_encodings_by_image_unique_flag[image_path] = True
    

    for image_path1, encodings1 in face_encodings_by_image.items():
        for image_path2, encodings2 in face_encodings_by_image.items():
            # print("working or not")
            if image_path1 == image_path2:
                continue
            try:
                distances = compare_faces(encodings1, encodings2)
            except:
                # print("yoooo")
                continue
            if np.min(distances) < 0.6:
               if(image_path1 in face_encodings_by_image_test and image_path2 in face_encodings_by_image_test):
                   del face_encodings_by_image_test[image_path2]
    
    # print(f"face values : {face_encodings_by_image_test.keys()}")

    # print(f"length of keys : {len(face_encodings_by_image_test)}")


    # for image_path1, encodings1 in face_encodings_by_image.items():
        # for image_path2, encodings2 in face_encodings_by_image.items():
        #     if image_path1 == image_path2:
        #         continue
        #     try:
        #         distances = compare_faces(encodings1, encodings2)
        #     except:
        #         continue
        #     # print(f"min distance {np.min(distances)} for imagepaths : {image_path1} and {image_path2}")
        #     if np.min(distances) > 0.6:  # You can adjust this threshold as needed
        #         if face_encodings_by_image_unique_flag[image_path1] == True and  face_encodings_by_image_unique_flag[image_path2] == True:
        #             print("both true")
        #             print(f"min distance {np.min(distances)} for imagepaths : {image_path1} and {image_path2}")
        #             unique_faces.add(image_path1)
        #             unique_faces.add(image_path2)
        #             face_encodings_by_image_unique_flag[image_path1] = False
        #             face_encodings_by_image_unique_flag[image_path2] = False
        #             print(f"unique_faces--- : {unique_faces}")
        #         elif face_encodings_by_image_unique_flag[image_path1] == False:
        #             if face_encodings_by_image_unique_flag[image_path2] == True:
        #                 print(f"{image_path2} true")
        #                 print(f"min distance {np.min(distances)} for imagepaths : {image_path1} and {image_path2}")
        #                 unique_faces.add(image_path2)
        #                 face_encodings_by_image_unique_flag[image_path2] = False
        #                 print(f"unique_faces--- : {unique_faces}")
        #         elif face_encodings_by_image_unique_flag[image_path2] == False:
        #             if face_encodings_by_image_unique_flag[image_path1] == True:
        #                 print(f"{image_path1} true")
        #                 print(f"min distance {np.min(distances)} for imagepaths : {image_path1} and {image_path2}")
        #                 unique_faces.add(image_path1)
        #                 face_encodings_by_image_unique_flag[image_path1] = False
        #                 print(f"unique_faces--- : {unique_faces}")



        
    # print(f"unique faces {unique_faces}")
    return len(face_encodings_by_image_test)


def find_unique_faces2(image_paths):
    face_encodings_by_image = {}
    for image_path in image_paths:
        face_encodings = get_face_encodings(image_path)
        face_encodings_by_image[image_path] = face_encodings

    unique_faces = set()
    all_face_encodings = [encodings for encodings_list in face_encodings_by_image.values() for encodings in encodings_list]

    for image_path, encodings in face_encodings_by_image.items():
        for encoding in encodings:
            is_unique = True
            for other_encoding in all_face_encodings:
                if encoding == other_encoding:
                    continue
                try:
                    min_distance = np.min(compare_faces([encoding], [other_encoding]))
                except:
                    continue
                if min_distance < 0.6:
                    is_unique = False
                    break
            if is_unique:
                unique_faces.add(tuple(encoding))

    return len(unique_faces)

 
def find_duplicate_faces(image_paths):

    face_encodings_by_image = {}

    for image_path in image_paths:

        face_encodings = get_face_encodings(image_path)

        face_encodings_by_image[image_path] = face_encodings

    duplicate_faces = defaultdict(list)

    for image_path1, encodings1 in face_encodings_by_image.items():

        for image_path2, encodings2 in face_encodings_by_image.items():

            if image_path1 == image_path2:

                continue

            try:
                distances = compare_faces(encodings1, encodings2)
            except:
                continue

            if np.min(distances) < 0.6:  # You can adjust this threshold as needed

                duplicate_faces[image_path1].append(image_path2)

    return duplicate_faces

def list_image_files(folder_path):
    image_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith((".jpg", ".jpeg", ".png", ".bmp")):
            image_files.append('media/'+file_name)
    return image_files


def getFiles(path):
  files = list()
  if os.path.isdir(path):
    dirFiles = os.listdir(path)
    for file in dirFiles:
      filePath = os.path.join(path, file)
      if os.path.isdir(filePath):
        files = files + getFiles(filePath)
      else:
        kind = ft.guess(filePath)
        basename = os.path.basename(filePath)
        files.append({
          'dir': os.path.abspath(path),
          'path': filePath,
          'mime': None if kind == None else kind.mime,
          'filename': os.path.splitext(basename)[0]
        })
  else:
    kind = ft.guess(path)
    basename = os.path.basename(path)
    files.append({
      'dir': os.path.abspath(os.path.dirname(path)),
      'path': path,
      'mime': None if kind == None else kind.mime,
      'filename': os.path.splitext(basename)[0]
    })
              
  return files

def main(args):
  # total = 0
  
  # for file_name in image_files:
  #     print(file_name)

  # image_paths = ["image5.jpg", "image6.jpg","image7.jpg"]
  # duplicate_faces = find_duplicate_faces(image_files)
  # for image_path, duplicates in duplicate_faces.items():

  #     print(f"Duplicate faces found in {image_path}: {duplicates}")


  # Example usage

  folder_path = "media"
  image_files = list_image_files(folder_path)
  # print("List of image files:")

  print("Processing detected faces...")

  unique_faces_count = find_unique_faces(image_files)


  print(f"Count of unique faces: {unique_faces_count}")

  # print("[INFO] found {} face(s)".format(total))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  # options
  parser.add_argument("-i", "--input", required=True, help="path to input directory or file")
  parser.add_argument("-o", "--output", default="output/", help="path to output directory")
  parser.add_argument("-s", "--frames", default=30, help="skip frames")
  parser.add_argument("-p", "--padding", default=1.0, help="padding ratio around the face (default: 1.0)")
  
  args = vars(parser.parse_args())
  main(args)
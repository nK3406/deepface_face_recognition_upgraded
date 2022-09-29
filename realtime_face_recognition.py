#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 10:53:47 2022

@author: orhan
"""

import os
import argparse
from turtle import distance
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import time
import re
import matplotlib as plt
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from deepface.basemodels import VGGFace, OpenFace, Facenet, Facenet512, FbDeepFace, DeepID, DlibWrapper, ArcFace, Boosting
from deepface import DeepFace
from deepface.extendedmodels import Age
from deepface.commons import functions, realtime, distance as dst
from deepface.detectors import FaceDetector





def realtime_face_recognition(db_path, 
                              model_name = "Facenet512", 
                              detector_backend = "opencv", 
                              distance_metric = "euclidean_l2", 
                              enable_face_analysis = True, 
                              source = 0, 
                              frame_threshold = 30,  
                              recognition_sensitivity = 0.8,
                              smile_sensitivity = 0.012):



    #------------------------ 
    face_detector = FaceDetector.build_model(detector_backend) 

    # Modeli build ediyoruz. (yüz tespiti için)
    print("Detector backend is ", detector_backend)
        
        #------------------------
        
    input_shape = (224, 224); input_shape_x = input_shape[0]; input_shape_y = input_shape[1] # yüzün croplanmış halinin girdi boyutları

    text_color = (255,255,255)

    employees = [] # Database'imizdeki kişilerin fotoğrafları
    print(db_path)
        
        #check passed db folder exists
    if os.path.isdir(db_path) == True:
            for r, d, f in os.walk(db_path): # r=root, d=directories, f = files
                for file in f:
                    if ('.jpg' in file):
                            #exact_path = os.path.join(r, file)
                                exact_path = r + "/" + file
                            #print(exact_path)
                                employees.append(exact_path)
                        
                        
    if len(employees) == 0:
            print("WARNING: There is no image in this path ( ", db_path,") . Face recognition will not be performed.")

        #------------------------

        # Eğer ki kişilerin fotoğrafı mevcutsa yüzü vektör formuna değiştirecek modeli build ediyoruz
    if len(employees) > 0:

            model = DeepFace.build_model(model_name) # yüzü vektör formuna değiştirecek modeli build ediyoruz
            print(model_name," is built")

            #------------------------

            input_shape = functions.find_input_shape(model)
            input_shape_x = input_shape[0]; input_shape_y = input_shape[1]

            #tuned thresholds for model and metric pair
            threshold = dst.findThreshold(model_name, distance_metric)

        #------------------------
        
        
        #facial attribute analysis models

    if enable_face_analysis == True:

            tic = time.time()

            emotion_model = DeepFace.build_model('Emotion')
            print("Emotion model loaded")

            age_model = DeepFace.build_model('Age')
            print("Age model loaded")

            gender_model = DeepFace.build_model('Gender')
            print("Gender model loaded")

            toc = time.time()

            print("Facial attibute analysis models loaded in ",toc-tic," seconds")

        #------------------------

        #find embeddings for employee list

    tic = time.time()

        #-----------------------

    pbar = tqdm(range(0, len(employees)), desc='Finding embeddings')

    #-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------
    #-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------
    #-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------


    model_names = ["Facenet512"]
    file_name = "representations_%s_%s.pkl" %(model_name,detector_backend) 
    file_name = file_name.replace("-", "_").lower()
    employees = []
    representations = []
    detect_indx = False
    enforce_detection = False

    if os.path.isdir(db_path) == True:
            model = DeepFace.build_model(model_name)
            models = {}
            models[model_name] = model
            models = {}
            models[model_name] = model
            if os.path.exists(db_path+"/"+file_name):
                f = open(db_path+'/'+file_name, 'rb')
                representations = pickle.load(f)
            else:
                for r, d, f in os.walk(db_path): # r=root, d=directories, f = files
                    for file in f:
                        if ('.jpg' in file.lower()) or ('.png' in file.lower()):
                            exact_path = r + "/" + file
                            employees.append(exact_path)
                if len(employees) == 0:
                    raise ValueError("There is no image in ", db_path," folder! Validate .jpg or .png files exist in this db_path.")  
                    #------------------------
                    #find representations for db images
                
                pbar = tqdm(range(0,len(employees)), desc='Finding representations')
        
        #for employee in employees:
                for index in pbar:
                        employee = employees[index]
        
                        instance = []
                        instance.append(employee)
        
                        for j in model_names:
                            representation = DeepFace.represent(img_path = employee
                                , model = models[j]
                                , enforce_detection = False
                                , detector_backend = detector_backend
                                )
        
                            instance.append(representation)
        
                        #-------------------------------
        
                            representations.append(instance)
    
                f = open(db_path+'/'+file_name, "wb")
                pickle.dump(representations, f)
                f.close()


    embeddings = representations

    #-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------
    #-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------
    #-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------
    #-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------#-------------------------------





    df = pd.DataFrame(embeddings, columns = ['employee', 'embedding']) # kişinin fotoğrafında yüzünün vektöre dönüştürülmüş halini içeren bir dataframe'imiz var artık.
    df['distance_metric'] = distance_metric

    toc = time.time()

    print("Embeddings found for given data set in ", toc-tic," seconds")
        
        #-----------------------
        
    pivot_img_size = 112 #face recognition result image

        #-----------------------

    freeze = False
    face_detected = False
    face_included_frames = 0 #freeze screen if face detected sequantially 5 frames
    freezed_frame = 0
    tic = time.time()
    employee_name_list = []


    cap = cv2.VideoCapture(source) #webcam
        
        
    while(True):
            ret, img = cap.read()
            if img is None:
                break
            
            raw_img = img.copy()
            resolution = img.shape; resolution_x = img.shape[1]; resolution_y = img.shape[0]
            
            try: # frame'de yüz olup olmadığı kontrol
                    #faces store list of detected_face and region pair
                    faces = FaceDetector.detect_faces(face_detector, detector_backend, img, align = False)
            except: #to avoid exception if no face detected
                    faces = []                
            if len(faces) == 0 or face_included_frames == frame_threshold:
                    face_included_frames =  0
                    
            detected_faces = []
            face_index = 0
            for face, (x, y, w, h) in faces:
                if w > resolution_x/5 and h < resolution_y - 40 and 0.5 < h/w < 2: #discard small detected faces
                    face_detected = True
                    if face_index == 0:
                        face_included_frames = face_included_frames + 1 #increase frame for a single face
                    cv2.rectangle(img, (x,y), (x+w,y+h), (67,67,67), 1) #draw rectangle to main image

                    # cv2.putText(img, str(frame_threshold - face_included_frames), (int(x+w/4),int(y+h/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)

                    detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face

                    #-------------------------------------

                    detected_faces.append((x,y,w,h))
                    face_index = face_index + 1
                    
                    
                    #-------------------------------------
                    
            if face_detected == True and face_included_frames == frame_threshold:  
                base_img = img.copy()
                # base_img = raw_img.copy()
                detected_faces = detected_faces.copy()
                tic = time.time()
                for detected_face in detected_faces:
                    x = detected_face[0]; y = detected_face[1]
                    w = detected_face[2]; h = detected_face[3]
                    
        
                    # cv2.rectangle(img, (x,y), (x+w,y+h), (67,67,67), 1) #draw rectangle to main image
        
                                #-------------------------------
        
                                #apply deep learning for custom_face
        
                    custom_face = base_img[y:y+h, x:x+w]
        
                                #-------------------------------
                                #facial attribute analysis
        
                    if enable_face_analysis == True:
                        tic = time.time()
                        gray_img = functions.preprocess_face(img = custom_face, target_size = (48, 48), grayscale = True, enforce_detection = enforce_detection, detector_backend = detector_backend)
                        emotion_labels = ['Happy']
                        emotion_predictions = emotion_model.predict(gray_img)[0,:]
                        sum_of_predictions = emotion_predictions.sum()
        
                        mood_items = []
                        for i in range(0, len(emotion_labels)):
                            mood_item = []
                            emotion_label = emotion_labels[i]
                            emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
                            mood_item.append(emotion_label)
                            mood_item.append(emotion_prediction)
                            mood_items.append(mood_item)
        
                        emotion_df = pd.DataFrame(mood_items, columns = ["emotion", "score"])
                        emotion_df = emotion_df.sort_values(by = ["score"], ascending=False).reset_index(drop=True)
        
                                    #background of mood box
        
                                    #transparency
                        overlay = img.copy()
                        opacity = 0.4
            # AGE --- GENDER PREDICTION

            #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------


                        # face_224 = functions.preprocess_face(img = custom_face, target_size = (224, 224), grayscale = False, enforce_detection = False, detector_backend = 'opencv')
        
            #          age_predictions = age_model.predict(face_224)[0,:]
            #          apparent_age = Age.findApparentAge(age_predictions)
        
                                    # #-------------------------------
        
            #          gender_prediction = gender_model.predict(face_224)[0,:]
        
            #          if np.argmax(gender_prediction) == 0:
            #                          gender = "W"
            #          elif np.argmax(gender_prediction) == 1:
            #                          gender = "M"
        
                                    # #print(str(int(apparent_age))," years old ", dominant_emotion, " ", gender)

            #          analysis_report = str(int(apparent_age))+" "+gender

            #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                        toc = time.time()
                        gzaman = toc-tic
                                    
                    custom_face = functions.preprocess_face(img = custom_face, target_size = (input_shape_y, input_shape_x), enforce_detection = enforce_detection, detector_backend = detector_backend)
                                    #check preprocess_face function handled
                    if custom_face.shape[1:3] == input_shape:
                                        if df.shape[0] > 0: #if there are images to verify, apply face recognition
                                            img1_representation = model.predict(custom_face)[0,:]
                                            tic = time.time()

                                            def findDistance(row):
                                                            distance_metric = row['distance_metric']
                                                            img2_representation = row['embedding']
                                                            distance = 1000 #initialize very large value
                                                            if distance_metric == 'cosine':
                                                                distance = dst.findCosineDistance(img1_representation, img2_representation)
                                                            elif distance_metric == 'euclidean':
                                                                distance = dst.findEuclideanDistance(img1_representation, img2_representation)
                                                            elif distance_metric == 'euclidean_l2':
                                                                distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation))
                                                            return distance
                                            df['distance'] = df.apply(findDistance, axis = 1)
                                            df = df.sort_values(by = ["distance"])
                                            zaman = time.time() - tic
                                            candidate = df.iloc[0]
                                            employee_name = candidate['employee']
                                            best_distance = candidate['distance']
                                            emo_score_best = emotion_df["score"].to_string()
                                            emo_score_best = emo_score_best.replace(" ","")
                                            emo_score_best = float(emo_score_best)                          
                                            #if True:
                                            if best_distance <= recognition_sensitivity:
                                                print("founded face :%s" %(employee_name))
                                                display_img = cv2.imread(employee_name)
                                                detect_indx = True
                                                display_img = cv2.resize(display_img, (pivot_img_size, pivot_img_size))
                                                label = employee_name.split("/")[-2].replace(".jpg", "")
                                                label = re.sub('[0-9]', '', label)
                                            else:
                                                display_img = None
                                                detect_indx = False
            try:
                if emo_score_best > smile_sensitivity:
                    cv2.putText(img,"LUTFEN IYICE GULUMSEYINIZ :) ", (90,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                else:
                    cv2.putText(img,"TESEKKURLER :)", (112,400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

                if detect_indx:
                    # try:      
                                overlay = img.copy()
                                opacity = 0.4
                                # if x+w+pivot_img_size < resolution_x:
                                #                 #right
                                #                 cv2.rectangle(img
                                #                     #, (x+w,y+20)
                                #                     , (x+w,y)
                                #                     , (x+w+pivot_img_size, y+h)
                                #                     , (64,64,64),cv2.FILLED)
                
                                #                 cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
                
                                # elif x-pivot_img_size > 0:
                                #                 #left
                                #                 cv2.rectangle(img
                                #                     #, (x-pivot_img_size,y+20)
                                #                     , (x-pivot_img_size,y)
                                #                     , (x, y+h)
                                #                     , (64,64,64),cv2.FILLED)
                
                                #                 cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
                
                                # for index, instance in emotion_df.iterrows():
                                #     emotion_label = "%s " % (instance['emotion'])
                                #     emotion_score = instance['score']/100
                
                                #     bar_x = 35 #this is the size if an emotion is 100%
                                #     bar_x = int(bar_x * emotion_score)
                
                                #     if x+w+pivot_img_size < resolution_x:
                                #         text_location_y = y + 20 + (index+1) * 20
                                #         text_location_x = x+w
                                #         if text_location_y < y + h:
                                #                 cv2.putText(img, emotion_label, (text_location_x, text_location_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                #                 cv2.rectangle(img
                                #                             , (x+w+70, y + 13 + (index+1) * 20)
                                #                             , (x+w+70+bar_x, y + 13 + (index+1) * 20 + 5)
                                #                             , (255,255,255), cv2.FILLED)
                
                                #     elif x-pivot_img_size > 0:
                                #             text_location_y = y + 20 + (index+1) * 20
                                #             text_location_x = x-pivot_img_size
                
                                #             if text_location_y <= y+h:
                                #                 cv2.putText(img, emotion_label, (text_location_x, text_location_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                                #                 cv2.rectangle(img
                                #                             , (x-pivot_img_size+70, y + 13 + (index+1) * 20)
                                #                             , (x-pivot_img_size+70+bar_x, y + 13 + (index+1) * 20 + 5)
                                #                             , (255,255,255), cv2.FILLED)
                
                                            #-------------------------------
                
                                
                                            #-------------------------------
                
                                # info_box_color = (46,200,255)
                
                                            #top
                    #          if y - pivot_img_size + int(pivot_img_size/5) > 0:
                
                    #                          triangle_coordinates = np.array( [
                                            # 		(x+int(w/2), y)
                                            # 		, (x+int(w/2)-int(w/10), y-int(pivot_img_size/3))
                                            # 		, (x+int(w/2)+int(w/10), y-int(pivot_img_size/3))
                                            # 	] )
                
                    #                          cv2.drawContours(img, [triangle_coordinates], 0, info_box_color, -1)
                
                    #                          cv2.rectangle(img, (x+int(w/5), y-pivot_img_size+int(pivot_img_size/5)), (x+w-int(w/5), y-int(pivot_img_size/3)), info_box_color, cv2.FILLED)
                
                    #                          cv2.putText(img, analysis_report, (x+int(w/3.5), y - int(pivot_img_size/2.1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)
                
                                            # #bottom
                    #          elif y + h + pivot_img_size - int(pivot_img_size/5) < resolution_y:
                
                    #                          triangle_coordinates = np.array( [
                                            # 		(x+int(w/2), y+h)
                                            # 		, (x+int(w/2)-int(w/10), y+h+int(pivot_img_size/3))
                                            # 		, (x+int(w/2)+int(w/10), y+h+int(pivot_img_size/3))
                                            # 	] )
                
                    #                          cv2.drawContours(img, [triangle_coordinates], 0, info_box_color, -1)
                
                    #                          cv2.rectangle(img, (x+int(w/5), y + h + int(pivot_img_size/3)), (x+w-int(w/5), y+h+pivot_img_size-int(pivot_img_size/5)), info_box_color, cv2.FILLED)
                
                    #                          cv2.putText(img, analysis_report, (x+int(w/3.5), y + h + int(pivot_img_size/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 111, 255), 2)

                                if y - pivot_img_size > 0 and x + w + pivot_img_size < resolution_x:
                                                                # top right
                                                                img[y - pivot_img_size:y, x+w:x+w+pivot_img_size] = display_img
                    
                                                                overlay = img.copy(); opacity = 0
                                                                cv2.rectangle(img,(x+w,y),(x+w+pivot_img_size, y+20),(46,200,255),cv2.FILLED)
                                                                cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
                    
                                                                cv2.putText(img, label, (x+w, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                                                                #connect face and text
                                                                cv2.line(img,(x+int(w/2), y), (x+3*int(w/4), y-int(pivot_img_size/2)),(67,67,67),1)
                                                                cv2.line(img, (x+3*int(w/4), y-int(pivot_img_size/2)), (x+w, y - int(pivot_img_size/2)), (67,67,67),1)
                                                                
                                elif y + h + pivot_img_size < resolution_y and x - pivot_img_size > 0:
                                                                #bottom left
                                                                img[y+h:y+h+pivot_img_size, x-pivot_img_size:x] = display_img
                    
                                                                overlay = img.copy(); opacity = 0.4
                                                                cv2.rectangle(img,(x-pivot_img_size,y+h-20),(x, y+h),(46,200,255),cv2.FILLED)
                                                                cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
                    
                                                                cv2.putText(img, label, (x - pivot_img_size, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                                                                #connect face and text
                                                                cv2.line(img,(x+int(w/2), y+h), (x+int(w/2)-int(w/4), y+h+int(pivot_img_size/2)),(67,67,67),1)
                                                                cv2.line(img, (x+int(w/2)-int(w/4), y+h+int(pivot_img_size/2)), (x, y+h+int(pivot_img_size/2)), (67,67,67),1)
                    
                                elif 280 > y - pivot_img_size > 0 and 280 > x - pivot_img_size > 0:
                                                                #top left
                                                                img[y-pivot_img_size:y, x-pivot_img_size:x] = display_img
                    
                                                                overlay = img.copy(); opacity = 0.4
                                                                cv2.rectangle(img,(x- pivot_img_size,y),(x, y+20),(46,200,255),cv2.FILLED)
                                                                cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

                                                                cv2.putText(img, label, (x - pivot_img_size, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                    
                                                                #connect face and text
                                                                cv2.line(img,(x+int(w/2), y), (x+int(w/2)-int(w/4), y-int(pivot_img_size/2)),(67,67,67),1)
                                                                cv2.line(img, (x+int(w/2)-int(w/4), y-int(pivot_img_size/2)), (x, y - int(pivot_img_size/2)), (67,67,67),1)
                    
                                elif x+w+pivot_img_size < resolution_x and y + h + pivot_img_size < resolution_y:
                                                                #bottom right
                                                                img[y+h:y+h+pivot_img_size, x+w:x+w+pivot_img_size] = display_img
                                                                overlay = img.copy(); opacity = 0.4
                                                                cv2.rectangle(img,(x+w,y+h-20),(x+w+pivot_img_size, y+h),(46,200,255),cv2.FILLED)
                                                                cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
                                                                cv2.putText(img, label, (x+w, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                                                                
                                                                #connect face and text
                                                                cv2.line(img,(x+int(w/2), y+h), (x+int(w/2)+int(w/4), y+h+int(pivot_img_size/2)),(67,67,67),1)
                                                                cv2.line(img, (x+int(w/2)+int(w/4), y+h+int(pivot_img_size/2)), (x+w, y+h+int(pivot_img_size/2)), (67,67,67),1)
                                else:
                                                                img[336:448,0:112] = display_img 
                                                                overlay = img.copy(); opacity = 0.2
                                                                cv2.rectangle(img,(0,336),(112,318),(46,200,255),cv2.FILLED)
                                                                cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
                    
                                                                cv2.putText(img, label, (1, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                    
                                                                # #connect face and text
                                                                # cv2.line(img,(x+int(w/2), y), (x+3*int(w/4), y-int(pivot_img_size/2)),(67,67,67),1)
                                                                # cv2.line(img, (x+3*int(w/4), y-int(pivot_img_size/2)), (x+w, y - int(pivot_img_size/2)), (67,67,67),1)
            except Exception as err:
                        print(str(err))
            
                                    #-------------------------------                         
            

            cv2.imshow('img',img)

            if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
                break
        

        #kill open cv things
    cap.release()
    cv2.destroyAllWindows()
    return best_distance,emo_score_best,img,x,y

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str ,help='Database path')
    parser.add_argument('--model_name', type=str, default="Facenet512", help='Model of face recognition system.')
    parser.add_argument('--detector_backend', type=str, default="opencv", help='Detector-backend to detect face.')
    parser.add_argument('--distance_metric', type=str, default="euclidean_l2", help='Distance metric system of face recognition sys. Calculator of difference between vectors')
    parser.add_argument('--enable_face_analysis', type=str, default=True, help='Enable face analysis to find gestures. ')
    parser.add_argument('--source', type=int, default=0, help='Source of image, default = Webcam(0)')
    parser.add_argument('--frame_threshold', type=int, default=30, help='total number of face included frames to analyze')
    parser.add_argument('--recognition_sensitivity', type=float, default=0.8, help='Limit value of recognition, aka difference between vectors or faces, lower for high security and similarity, high values for lower security and similarity but higher recognition.')
    parser.add_argument('--smile_sensitivity',type=float, default=0.012, help='Limit value of smile parameter, higher values can lead to more wrong results but captures smaller gestures')
    opt = parser.parse_args()
    print(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    realtime_face_recognition(**vars(opt))





















                    
                    
                    

#module to define data analysis functions

import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
from IPython.display import HTML
import cv2
import csv
import pandas as pd

# Initializing mediapipe pose class.
# mp_pose = mp.solutions.pose

# # Setting up the Pose function.
# pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5,
#             min_tracking_confidence=0.5)
# # Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils 

def detectPose(image, pose, display=False):
    
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image, 
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''
    mp_pose = mp.solutions.pose
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:
    
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z )))
    
    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
    
        # Display the original input image and the resultant image.
        # plt.figure(figsize=[22,22])
        # plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        # plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        #draw in cv2
        cv2.imshow('Output Image', output_image)
        
        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        
    # Otherwise
    else:
        
        # Return the output image and the found landmarks.
        return output_image, landmarks


def calculateAngle(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.

    '''

    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Check if the angle is less than zero.
    if angle < 0:

        # Add 360 to the found angle.
        angle += 360
    
    # Return the calculated angle.
    return angle

#Calculate angles with 3d landmarks
def angle3d(lm1, lm2, lm3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.

    '''


    # Get the required landmarks coordinates.
    x1, y1, z1 = lm1
    x2, y2, z2 = lm2
    x3, y3, z3 = lm3

    #Calculate the angle between the three points with lm2 as the vertex
    v1 = np.array([x1 - x2, y1 - y2, z1 - z2])
    v2 = np.array([x3 - x2, y3 - y2, z3 - z2])
    v1 = v1 / np.linalg.norm(v1) if np.linalg.norm(v1) != 0 else v1
    v2 = v2 / np.linalg.norm(v2) if np.linalg.norm(v2) != 0 else v2
    angle = np.degrees(np.arccos(np.dot(v1, v2)))
    if angle < 0:

        # Add 360 to the found angle.
        angle += 360
    
    # Return the calculated angle.
    return angle
    

#Prepare joint angle dataset from photos

# Initializing mediapipe pose class.
def populate_row(img,label,test = False):
      mp_pose = mp.solutions.pose
      
      # Setting up the Pose function.
      pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5,
                  min_tracking_confidence=0.5)
      output_image, lm = detectPose(img, pose, display=False)
      # Initializing mediapipe drawing class, useful for annotation.
      #mediapipe gives 33 landmarks, create a list of all the joint angles possible
      #dict of all possible joint angles
      #handle missing landmarks
      
      joint_angles_dict = {}
      try:
            joint_angles_dict['left_elbow_angle'] = angle3d(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                            lm[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                                lm[mp_pose.PoseLandmark.LEFT_WRIST.value]) 
      except:
            joint_angles_dict['left_elbow_angle'] = None

      try:

      
            joint_angles_dict['right_elbow_angle'] = angle3d(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                      lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                          lm[mp_pose.PoseLandmark.RIGHT_WRIST.value])
      
      except:
            joint_angles_dict['right_elbow_angle'] = None
      
      try:
            joint_angles_dict['left_shoulder_angle'] = angle3d(lm[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                                      lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                          lm[mp_pose.PoseLandmark.LEFT_HIP.value])
      except:
            joint_angles_dict['left_shoulder_angle'] = None
      try:
            joint_angles_dict['right_shoulder_angle'] = angle3d(lm[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                      lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
      except:
            joint_angles_dict['right_shoulder_angle'] = None
      try:
            joint_angles_dict['left_knee_angle'] = angle3d(lm[mp_pose.PoseLandmark.LEFT_HIP.value],
                                                lm[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                          lm[mp_pose.PoseLandmark.LEFT_ANKLE.value])
      except:
            joint_angles_dict['left_knee_angle'] = None
      try:
            joint_angles_dict['right_knee_angle'] = angle3d(lm[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                      lm[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                          lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
           

      except:
            joint_angles_dict['right_knee_angle'] = None
      try:
            joint_angles_dict['left_hip_angle'] = angle3d(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                lm[mp_pose.PoseLandmark.LEFT_HIP.value],
                                                lm[mp_pose.PoseLandmark.LEFT_KNEE.value])
      except:
            joint_angles_dict['left_hip_angle'] = None
      try:
            joint_angles_dict['right_hip_angle'] = angle3d(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                lm[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                lm[mp_pose.PoseLandmark.RIGHT_KNEE.value])
      except:
            joint_angles_dict['right_hip_angle'] = None
      try:
            joint_angles_dict['left_ankle_angle'] = angle3d(lm[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                                      lm[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                                                lm[mp_pose.PoseLandmark.LEFT_HEEL.value])
      except:
            joint_angles_dict['left_ankle_angle'] = None
      try:
            joint_angles_dict['right_ankle_angle'] = angle3d(lm[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                                      lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                                                lm[mp_pose.PoseLandmark.RIGHT_HEEL.value])
      except:
            joint_angles_dict['right_ankle_angle'] = None


      #distance between two palms
      try:
            height, width, _ = img.shape
            palm_dist = np.array(lm[mp_pose.PoseLandmark.LEFT_WRIST.value])-np.array(lm[mp_pose.PoseLandmark.RIGHT_WRIST.value])
            if label=='tree':
                  #palms are in the same plane, so we only need 2d distance
                  palm_dist = np.array([palm_dist[0]/width,palm_dist[1]/height])
            else:
                  palm_dist = np.array([palm_dist[0]/width,palm_dist[1]/height,palm_dist[2]])
            joint_angles_dict['palm_distance'] = np.linalg.norm(palm_dist)
      except:
            joint_angles_dict['palm_distance'] = None
      #distance between two ankles
      try:
            height, width, _ = img.shape
            ankle_dist = np.array(lm[mp_pose.PoseLandmark.LEFT_ANKLE.value])-np.array(lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
            if label=='tree': 
                  #ankles are in the same plane, so we only need 2d distance
                  ankle_dist = np.array([ankle_dist[0]/width,ankle_dist[1]/height])
            else:
                  ankle_dist = np.array([ankle_dist[0]/width,ankle_dist[1]/height],ankle_dist[2])
            joint_angles_dict['ankle_distance'] = np.linalg.norm(ankle_dist)
      except:
            joint_angles_dict['ankle_distance'] = None
      
     
      #normalized vector from hip to head
      try:
            midpoint = (np.array(lm[mp_pose.PoseLandmark.LEFT_HIP.value])+np.array(lm[mp_pose.PoseLandmark.RIGHT_HIP.value]))/2
            v = np.array(lm[mp_pose.PoseLandmark.NOSE.value])-midpoint
            joint_angles_dict['hip_head_vector'] = v/np.linalg.norm(v)
      except:
            joint_angles_dict['hip_head_vector'] = None

      if label=='tree' or label=='warrior':
            lk = 0 if joint_angles_dict['left_knee_angle']==None else joint_angles_dict['left_knee_angle']
            rk = 0 if joint_angles_dict['right_knee_angle']==None else joint_angles_dict['right_knee_angle']
            lh = 0 if joint_angles_dict['left_hip_angle']==None else joint_angles_dict['left_hip_angle']
            rh = 0 if joint_angles_dict['right_hip_angle']==None else joint_angles_dict['right_hip_angle']
            la = 0 if joint_angles_dict['left_ankle_angle']==None else joint_angles_dict['left_ankle_angle']
            ra = 0 if joint_angles_dict['right_ankle_angle']==None else joint_angles_dict['right_ankle_angle']

            ls = 0 if joint_angles_dict['left_shoulder_angle']==None else joint_angles_dict['left_shoulder_angle']
            rs = 0 if joint_angles_dict['right_shoulder_angle']==None else joint_angles_dict['right_shoulder_angle']
            le = 0 if joint_angles_dict['left_elbow_angle']==None else joint_angles_dict['left_elbow_angle']
            re = 0 if joint_angles_dict['right_elbow_angle']==None else joint_angles_dict['right_elbow_angle']
            
            #Due to bad data, we need to filter out some poses
            if not test and ( ls < 90 or rs < 90 or joint_angles_dict['palm_distance']>0.3 ): #TODO : Correct this later
                  return

            #now due to unsymmetrical pose, we need to exchange the angles of the legs, left --> max, right --> min
            joint_angles_dict['left_knee_angle'] = lk if lk>rk else rk
            joint_angles_dict['right_knee_angle'] = rk if lk>rk else lk
            joint_angles_dict['left_hip_angle'] = lh if lh>rh else rh
            joint_angles_dict['right_hip_angle'] = rh if lh>rh else lh
            joint_angles_dict['left_ankle_angle'] = la if la>ra else ra
            joint_angles_dict['right_ankle_angle'] = ra if la>ra else la
            # joint_angles_dict['left_shoulder_angle'] = ls if ls>rs else rs
            # joint_angles_dict['right_shoulder_angle'] = rs if ls>rs else ls
            # joint_angles_dict['left_elbow_angle'] = le if le>re else re
            # joint_angles_dict['right_elbow_angle'] = re if le>re else le


      


      if test:
            return joint_angles_dict, lm, output_image
      

      
      #else populate a row
      row = []
      for key in joint_angles_dict:
            row.append(joint_angles_dict[key])
      row.append(label)
      #append to csv
      with open('joint_angles.csv', 'a',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
      #close csv
            f.close()

      

def create_ci(distribution,alpha=0.95):

    #calculate mean and std of each joint angle
    df = pd.read_csv('joint_angles.csv')
    df.describe()
    #find z value for given alpha using library
    from scipy import stats
    #z = stats.norm.ppf(1-(1-alpha)/2)
    z = 1.96
    #remove outliers
    #remove rows with any NaN values
    #remove rows with any value outside 4 std 
    # for col in df.columns:
    #       if col!='pose':
    #             df = df[np.abs(df[col]-df[col].mean()) <= (4*df[col].std())]


    #create diff dfs for each pose
    df_tree = df[df['label']=='tree']
    df_warrior = df[df['label']=='warrior']
    df_dog = df[df['label']=='downdog']
    df_poses = {'tree':df_tree, 'downdog':df_dog}
    for i in df_poses:
        for col in df_poses[i].columns:
                if col!='label'and col!='hip_head_vector':
                    df_poses[i] = df_poses[i][np.abs(df_poses[i][col]-df_poses[i][col].mean()) <= (4*df_poses[i][col].std())]
                # if col=='palm_distance' or col=='ankle_distance':
                #       df_poses[i] = df_poses[i][np.abs(df_poses[i][col]-df_poses[i][col].mean()) <= (4*df_poses[i][col].std())]
                    


    # #z score = (x - mean)/std
    # #Creating dict of confidence intervals
    ci = {}
    # #iterate over rows
    for i in df_poses:
        pose_ci = {}
        for col in df_poses[i].columns:
                #if column is not pose
                if col!='label' and col!='hip_head_vector':
                    #calculate mean and std
                    mean = df_poses[i][col].mean()
                    std = df_poses[i][col].std()
                    #z value for 95% confidence interval is 1.96
                    pose_ci[col] = [mean-z*std, mean+z*std]
                    print(i,col, pose_ci[col])
        ci[i] = pose_ci

    df_tree.describe()
            
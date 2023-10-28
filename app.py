import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# print(results.pose_landmarks)
# print(mp_pose.POSE_CONNECTIONS)
# print(landmarks)

def calculate_angle(a, b, c):
  a = np.array(a)
  b = np.array(b)
  c = np.array(c)
  radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
  angle = np.abs(radians*180.0/np.pi)
  if angle>180.0:
    angle = 360 - angle
  return angle

#--------------------------------------------------BICEP CURL-------------------------------------------

def bicep_curl():
  counter = 0
  stage = None

  cap = cv2.VideoCapture(0)
# set up multiple instance
  with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
      ret,frame = cap.read()
# recolor image to rgb
      image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      image.flags.writeable = False
# make detections
      results = pose.process(image)
# recolor image to bgr
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Extraction
      try:
        landmarks = results.pose_landmarks.landmark
      # print(landmarks)
# get coordinates
        lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        lhip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        rhip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

  # calculate angle
        langlesew = calculate_angle(lshoulder,lelbow, lwrist)
        ranglesew = calculate_angle(rshoulder,relbow, rwrist)
        lanmgle_hse = calculate_angle(lhip, lshoulder,lelbow)
        ranmgle_hse = calculate_angle(rhip, rshoulder,relbow)

  # visualize angle
        cv2.putText(image, str(langlesew),
                    tuple(np.multiply(lelbow,[640,480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,
                    cv2.LINE_AA)
        cv2.putText(image, str(ranglesew),
                    tuple(np.multiply(relbow,[640,480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,
                    cv2.LINE_AA)
        cv2.putText(image, str(lanmgle_hse),
                    tuple(np.multiply(lshoulder,[640,480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,
                    cv2.LINE_AA)
        cv2.putText(image, str(ranmgle_hse),
                    tuple(np.multiply(rshoulder,[640,480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,
                    cv2.LINE_AA)
  # Curl counter logic
        if langlesew > 160 and ranglesew > 160:
          stage = "down"
        if langlesew < 30 and ranglesew < 30 and stage =='down':
          stage = "up"
          counter = counter + 1
          # print(counter)
      except:
        pass

      cv2.rectangle(image, (0,0),(225, 73),(245,117, 16), -1)
      cv2.putText(image, 'REPS', (15,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
      cv2.putText(image, str(counter),(10,60),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)

  # render detections
      mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

      cv2.imshow("Mediapipe Feed", image)
      if cv2.waitKey(10) &0xFF == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()

# for lndmrk in mp_pose.PoseLandmark:
#   print(lndmrk.value, lndmrk.name)
# bicep_curl()

#--------------------------------------------------PUSH UP-------------------------------------------

def push_up():
  counter = 0
  stage = None

  cap = cv2.VideoCapture(0)
# set up multiple instance
  with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
      ret,frame = cap.read()
# recolor image to rgb
      image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      image.flags.writeable = False
# make detections
      results = pose.process(image)
# recolor image to bgr
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Extraction
      try:
        landmarks = results.pose_landmarks.landmark
      # print(landmarks)
# get coordinates
        lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        # lhip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        # rhip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

  # calculate angle
        langlesew = calculate_angle(lshoulder,lelbow, lwrist)
        ranglesew = calculate_angle(rshoulder,relbow, rwrist)
        # lanmgle_hse = calculate_angle(lhip, lshoulder,lelbow)
        # ranmgle_hse = calculate_angle(rhip, rshoulder,relbow)

  # visualize angle
        cv2.putText(image, str(langlesew),
                    tuple(np.multiply(lelbow,[640,480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,
                    cv2.LINE_AA)
        cv2.putText(image, str(ranglesew),
                    tuple(np.multiply(relbow,[640,480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,
                    cv2.LINE_AA)
        # cv2.putText(image, str(lanmgle_hse),
        #             tuple(np.multiply(lshoulder,[640,480]).astype(int)),
        #             cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,
        #             cv2.LINE_AA)
        # cv2.putText(image, str(ranmgle_hse),
        #             tuple(np.multiply(rshoulder,[640,480]).astype(int)),
        #             cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,
        #             cv2.LINE_AA)
  # Curl counter logic
        if langlesew < 100 and ranglesew < 100:
          stage = "down"
        if langlesew > 130 and ranglesew > 130 and stage =='down':
          stage = "up"
          counter = counter + 1
          # print(counter)
      except:
        pass

      cv2.rectangle(image, (0,0),(225, 73),(245,117, 16), -1)
      cv2.putText(image, 'REPS', (15,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
      cv2.putText(image, str(counter),(10,60),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)

  # render detections
      mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

      cv2.imshow("Mediapipe Feed", image)
      if cv2.waitKey(10) &0xFF == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()

# for lndmrk in mp_pose.PoseLandmark:
#   print(lndmrk.value, lndmrk.name)
# push_up()

#------------------------------------------------SQUATS---------------------------------------------------

def squats():
  counter = 0
  stage = None

  cap = cv2.VideoCapture(0)
# set up multiple instance
  with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
      ret,frame = cap.read()
# recolor image to rgb
      image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      image.flags.writeable = False
# make detections
      results = pose.process(image)
# recolor image to bgr
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Extraction
      try:
        landmarks = results.pose_landmarks.landmark
      # print(landmarks)
# get coordinates
        # lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        # rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        # lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        # relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        # lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        # rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        lhip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        rhip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        lknee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        rknee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        lankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        rankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]




  # calculate angle
        # langlesew = calculate_angle(lshoulder,lelbow, lwrist)
        # ranglesew = calculate_angle(rshoulder,relbow, rwrist)
        langle = calculate_angle(lhip, lknee,lankle)
        rangle = calculate_angle(rhip, rknee,rankle)

  # visualize angle
        # cv2.putText(image, str(langlesew),
        #             tuple(np.multiply(lelbow,[640,480]).astype(int)),
        #             cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,
        #             cv2.LINE_AA)
        # cv2.putText(image, str(ranglesew),
        #             tuple(np.multiply(relbow,[640,480]).astype(int)),
        #             cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,
        #             cv2.LINE_AA)
        cv2.putText(image, str(langle),
                    tuple(np.multiply(lknee,[640,480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,
                    cv2.LINE_AA)
        cv2.putText(image, str(rangle),
                    tuple(np.multiply(rknee,[640,480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,
                    cv2.LINE_AA)
  # Curl counter logic
        if langle <90 and rangle <90:
          stage = "down"
        if langle >160 and rangle >160 and stage =='down':
          stage = "up"
          counter = counter + 1
          # print(counter)
      except:
        pass

      cv2.rectangle(image, (0,0),(225, 73),(245,117, 16), -1)
      cv2.putText(image, 'REPS', (15,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
      cv2.putText(image, str(counter),(10,60),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)

  # render detections
      mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

      cv2.imshow("Mediapipe Feed", image)
      if cv2.waitKey(10) &0xFF == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()

squats()

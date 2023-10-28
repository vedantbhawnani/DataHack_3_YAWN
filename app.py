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

def bicep_curl(image, langlesew, lelbow, relbow, ranglesew):
  c1 = 0
  stage = None

  cap = cv2.VideoCapture(0)
# set up multiple instance
  while cap.isOpened():
      
        cv2.putText(image, str(langlesew),
                    tuple(np.multiply(lelbow,[640,480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,
                    cv2.LINE_AA)
        cv2.putText(image, str(ranglesew),
                    tuple(np.multiply(relbow,[640,480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,
                    cv2.LINE_AA)
  # Curl counter logic
        if langlesew > 160 and ranglesew > 160:
          stage = "down"
        if langlesew < 30 and ranglesew < 30 and stage =='down':
          stage = "up"
          c1 = c1 + 1

#--------------------------------------------------PUSH UP-------------------------------------------

def push_up(image, langlesew, ranglesew, lelbow, relbow):
  c2 = 0
  stage = None

  cap = cv2.VideoCapture(0)
# set up multiple instance
  while cap.isOpen():
    # visualize angle
    cv2.putText(image, str(langlesew),
                      tuple(np.multiply(lelbow,[640,480]).astype(int)),
                      cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,
                      cv2.LINE_AA)
    cv2.putText(image, str(ranglesew),
                      tuple(np.multiply(relbow,[640,480]).astype(int)),
                      cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,
                      cv2.LINE_AA)
    # Curl counter logic
    if langlesew < 100 and ranglesew < 100:
      stage = "down"
    if langlesew > 130 and ranglesew > 130 and stage =='down':
      stage = "up"
      c2 = c2 + 1
    cv2.rectangle(image, (0,0),(225, 73),(245,117, 16), -1, )
    cv2.putText(image, 'PUSH UP REPS', (15,12),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1,cv2.LINE_AA)
    cv2.putText(image, str(c2),(10,60),cv2.FONT_HERSHEY_SIMPLEX,1.7,(255,255,255),2,cv2.LINE_AA)

#------------------------------------------------SQUATS---------------------------------------------------

def squats(image, langle, lknee, rangle, rknee):
  c3 = 0
  stage = None

  cap = cv2.VideoCapture(0)
  while cap.isOpened():
      
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
        c3 = c3 + 1
      

def workout():
  # counter = 0
  # stage = None

  cap = cv2.VideoCapture(0)
  with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
      ret,frame = cap.read()
      image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      image.flags.writeable = False
      results = pose.process(image)
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      try:
        landmarks = results.pose_landmarks.landmark

        lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        lhip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        rhip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        lknee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        rknee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        lankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        rankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]


        langlesew = calculate_angle(lshoulder,lelbow, lwrist)
        ranglesew = calculate_angle(rshoulder,relbow, rwrist)
        langle = calculate_angle(lhip, lknee,lankle)
        rangle = calculate_angle(rhip, rknee,rankle)

        cv2.putText(image, str(langlesew),
                    tuple(np.multiply(lelbow,[640,480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,
                    cv2.LINE_AA)
        cv2.putText(image, str(ranglesew),
                    tuple(np.multiply(relbow,[640,480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,
                    cv2.LINE_AA)
        cv2.putText(image, str(langle),
                    tuple(np.multiply(lknee,[640,480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,
                    cv2.LINE_AA)
        cv2.putText(image, str(rangle),
                    tuple(np.multiply(rknee,[640,480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,
                    cv2.LINE_AA)
# curl counter
        if langlesew < 100 and ranglesew < 100:
          stage = "down"
          if langlesew > 130 and ranglesew > 130 and stage =='down':
            stage = "up"
            push_up(image, langlesew, ranglesew, lelbow, relbow)
        elif langle <90 and rangle <90:
          stage = "down"
          if langle >160 and rangle >160 and stage =='down':
            stage = "up"
            squats(image, langle, lknee, rangle, rknee)
        elif langlesew > 160 and ranglesew > 160:
          stage = "down"
          if langlesew < 30 and ranglesew < 30 and stage =='down':
            stage = "up"
            bicep_curl(image, langlesew, lelbow, relbow, ranglesew)
      except:
        pass

      # cv2.rectangle(image, (0,0),(225, 73),(245,117, 16), -1, )
      # cv2.putText(image, 'PUSH UP REPS', (15,12),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1,cv2.LINE_AA)
      # cv2.putText(image, str(c2),(10,60),cv2.FONT_HERSHEY_SIMPLEX,1.7,(255,255,255),2,cv2.LINE_AA)

      mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

      cv2.imshow("Mediapipe Feed", image)
      if cv2.waitKey(10) &0xFF == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()

workout()
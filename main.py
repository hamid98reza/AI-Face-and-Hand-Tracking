import cv2
import mediapipe
import numpy as np
import pandas as pd

# PUT YOUR FACE IN FRONT OF THE CAMERA TO GET A BLACK FRAME.😊
#⚠️BE CAREFUL ABOUT INDENTATIONS. 

mpDraw = mediapipe.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

mp_face_mesh = mediapipe.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

mp_hands = mediapipe.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       model_complexity=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

face_oval = mp_face_mesh.FACEMESH_FACE_OVAL
df = pd.DataFrame(list(face_oval), columns=["p1", "p2"])

handNo = 0 # hand number for counting with fingers
tipIds = [4, 8, 12, 16, 20] #Ids for tip of each finger according to mediapipe


cap = cv2.VideoCapture(0)

while True:
    true, img = cap.read()
    img = cv2.resize(img, (800, 600))
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    results2 = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    out = np.copy(img)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]

        routes_idx = []

        p1 = df.iloc[0]["p1"]
        p2 = df.iloc[0]["p2"]

        for i in range(0, df.shape[0]):
            obj = df[df["p1"] == p2]
            p1 = obj["p1"].values[0]
            p2 = obj["p2"].values[0]

            route_idx = []
            route_idx.append(p1)
            route_idx.append(p2)
            routes_idx.append(route_idx)

        routes = []

        for source_idx, target_idx in routes_idx:
            source = landmarks.landmark[source_idx]
            target = landmarks.landmark[target_idx]

            relative_source = (int(img.shape[1] * source.x), int(img.shape[0] * source.y))
            relative_target = (int(img.shape[1] * target.x), int(img.shape[0] * target.y))

            cv2.line(img, relative_source, relative_target, (255, 255, 255), thickness=2)

            routes.append(relative_source)
            routes.append(relative_target)

        for faceLms in results.multi_face_landmarks:  # comment these two lines two see your face on the black screen.
            mpDraw.draw_landmarks(img, faceLms, mp_face_mesh.FACEMESH_CONTOURS, drawSpec, drawSpec)

        mask = np.zeros((img.shape[0], img.shape[1]))
        mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
        mask = mask.astype(bool)

        out = np.zeros_like(img)
        out[mask] = img[mask]

    point_list = []
    if results2.multi_hand_landmarks:
        myHand = results2.multi_hand_landmarks[handNo]
        for hand_landmarks in results2.multi_hand_landmarks:
            mpDraw.draw_landmarks(out, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
                cv2.circle(out, (x, y), 5, (0, 0, 255), -1)


        for id, lm in enumerate(myHand.landmark):
            # print(id, lm)
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            point_list.append([id,cx,cy])
        zero_one = []
        if point_list != 0:

            if point_list[4][1] < point_list[4 - 1][1]:  
                zero_one.append(0) ## <-
            else:
                zero_one.append(1) ## <-

                    # 4 Fingers
            
            for i in range(1, 5):
                if point_list[tipIds[i]][2] < point_list[tipIds[i] - 2][2]:
                    zero_one.append(1)
                else:
                    zero_one.append(0)
                    
            cv2.putText(out, str(zero_one.count(1)), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 25)

        
    cv2.imshow("Out", out)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
import mediapipe as mp
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.models import Sequential,Model
from keras._tf_keras.keras.layers import LSTM,Dense,Input,Dropout,Flatten,Activation,RepeatVector,Multiply,BatchNormalization,Bidirectional,Conv1D,Permute,Lambda
from keras._tf_keras.keras.callbacks import TensorBoard
import time

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmark(image,results):
    # mp_drawing.draw_landmarks(image,results.face_landmarks,mp_holistic.FACEMESH_CONTOURS,mp_drawing.DrawingSpec(color=(80,110,10),thickness=4,circle_radius=1),mp_drawing.DrawingSpec(color=(150,1,130),thickness=5,circle_radius=4))
    mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(10,255,10),thickness=5,circle_radius=10),mp_drawing.DrawingSpec(color=(10,2,130),thickness=5,circle_radius=10))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def extract_keypoints(results):
    pose=np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh=np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh=np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten()if results.right_hand_landmarks else np.zeros(21*3)
    # face=np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    return np.concatenate([pose,lh,rh])




DATA_PATH = "data3"
asl_words = [
    'hello',
    'yes',
    'no',
    'thank you',
    'please',
    'sorry',
    'stop',
    'help',
    'what',
    'how',
    'where',
    'when',
    "eat",
    "drink",
    "sleep",
    "goodbye"
]
# asl_words=['hello','iloveyou','thanks']
actions = np.array(asl_words)
label_map={label:num for num,label in enumerate(actions)}
no_sequences = 50
seq_length = 60

def create_bins():
    for action in actions:
        for seq in range(no_sequences):
            dir_path = os.path.join(DATA_PATH, action, str(seq))
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created: {dir_path}")

#
# def data_collection():
#     cap = cv2.VideoCapture(0)
#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         for action in actions:
#             cv2.putText(image, f'WORD CHANGE TO {action}', (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 10,
#                         cv2.LINE_AA)
#             cv2.waitKey(5000)
#             for sequence in range(no_sequences):
#                 for frame_num in range(seq_length):
#                     ret, frame = cap.read()
#                     frame = cv2.flip(frame, 1)
#                     image, results = mediapipe_detection(frame, holistic)
#                     print(results)
#                     draw_landmark(image,results)
#                     if frame_num == 0:
#                         cv2.putText(image,'STARTING COLLECTION', (120, 200),cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10,cv2.LINE_AA)
#                         cv2.waitKey(2000)
#
#                     cv2.putText(image,'Collecting frames for {} Video Number {}'.format(action, sequence), (30, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6, cv2.LINE_AA)
#                     keypoints = extract_keypoints (results)
#                     npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
#                     np.save(npy_path, keypoints)
#                     cv2.imshow("OpenCV Feed", image)
#                     if cv2.waitKey(10) & 0xFF == ord('q'):
#                         break
#     cap.release()
#     cv2.destroyAllWindows()
#     plt.imshow(frame)


def data_collection2():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Camera couldn't be opened.")
        return

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            print(f"Starting collection for action: {action}")
            start_time = time.time()

            while time.time() - start_time < 5:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                frame=cv2.resize(frame,[1280,720])
                cv2.putText(frame, f'WORD CHANGE TO {action}', (100, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            for sequence in range(no_sequences):
                print(f"  Sequence {sequence + 1}/{no_sequences}")
                for frame_num in range(seq_length):
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    frame = cv2.flip(frame, 1)
                    image, results = mediapipe_detection(frame, holistic)
                    draw_landmark(image, results)

                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (80, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(1000)

                    cv2.putText(image, f'Collecting {action} | Seq {sequence}', (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)

                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    cv2.imshow("OpenCV Feed", image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
    cap.release()
    cv2.destroyAllWindows()
sequences,labels=[],[]

def sequencing():
    for action in actions:
        for sequence in range(no_sequences):
            video=[]
            for frame_num in range(seq_length):
                res=np.load(os.path.join(DATA_PATH,action,str(sequence),f"{frame_num}.npy"))
                video.append(res)
            sequences.append(video)
            labels.append(label_map[action])
    print(np.array(sequences).shape)
    print(np.array(labels).shape)

def build_hybrid_model(input_shape=(60, 258), num_classes=4):  # Update num_classes if needed
    inputs = Input(shape=input_shape)

    # Conv1D block
    x = Conv1D(256, kernel_size=3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # BiLSTM block
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)

    # Attention mechanism
    attention = Dense(1, activation='tanh')(x)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(128)(attention)
    attention = Permute([2, 1])(attention)
    x = Multiply()([x, attention])
    x = Lambda(lambda z: tf.reduce_sum(z, axis=1), output_shape=(128,))(x)

    # Dense layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model
model = build_hybrid_model((60,258),actions.shape[0])
# model=Sequential()
# model.add(Input((30,1662)))
# model.add(LSTM(64,return_sequences=True,activation='relu'))
# model.add(LSTM(128,return_sequences=True,activation='relu'))
# model.add(LSTM(256,return_sequences=True,activation='relu'))
# model.add(LSTM(128,return_sequences=True,activation='relu'))
# model.add(LSTM(64,return_sequences=False,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(32,activation='relu'))
# model.add(Dense(actions.shape[0],activation='softmax'))
#
model.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=['categorical_accuracy'])

import dlib, cv2
import numpy as np

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

#descs = np.load('descs.npy)()

def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def find_faces(img):
    dets = detector(img, 1) #얼굴 찾은 결과물 들어감.

    if len(dets) == 0:
        return np.empty(0), np.empty(0), np.empty(0)

    rects, shapes = [], [] #얼굴 랜드마크 만들기 68개 점
    shapes_np = np.zeros((len(dets), 68, 2), dtype = np.int)

    for k,d in enumerate(dets): #얼굴 찾은 갯수 만큼 루프
        rect = ( (d.left(), d.top()), (d.right(), d.bottom()) )
        rects.append(rect)

        shape = sp(img, d)

        for i in range(0, 68):
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

        shapes.append(shape)

    return rects, shapes, shapes_np

def encode_faces(img, shapes): #얼굴 -> 128개의 벡터로 변환
    face_descriptors = []

    for shape in shapes:
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        #전체 이미지랑, 랜드마크 입력해서 인코딩
        face_descriptors.append(np.array(face_descriptor)) #array로 변환

    return np.array(face_descriptors)

def face_position(img):
    dets = detector(img, 1)

    if len(dets) == 0:
        return np.empty(0)

    for k, d in enumerate(dets):
        print('Detection {}: LEFT: {} TOP: {} RIGHT: {} BOTTOM: {}'.format(k+1, d.left(), d.top(), d.right(), d.bottom()))
        shape = sp(img, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)

        return np.array(face_descriptor)


img_paths = {
    'jihun' : 'images/jihun.JPG',
    'sung' : 'images/sung.jpeg' ,
    'joon' : 'images/joon.JPG'
}

descs = {
    'jihun' : None,
    'sung' : None ,
    'joon' : None
}

for name, img_path in img_paths.items():
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    _, img_shapes, _ = find_faces(img_rgb) #landmark를 받아서
    descs[name] = encode_faces(img_rgb, img_shapes)[0]




cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    exit()

while True:
    ret, img = cap.read()
    if not ret:
        print('not video')
        break

    img_bgr = img
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    rects, shapes, _ = find_faces(img_rgb)
    img_encoded = encode_faces(img_rgb, shapes)

    print('rects',rects)


    if len(img_encoded) == 0:
        continue

    for i, desc in enumerate(img_encoded):
        found = False
        for name, saved_desc in descs.items():
            dist = np.linalg.norm([desc] - saved_desc, axis = 1)

            if dist < 0.4:
                found = True

                cv2.rectangle(img_rgb, rects[i][0], rects[i][1], color = (255, 255, 255), thickness=2)
                cv2.putText(img_rgb, name, org=(rects[i][1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color= (255,255,255), thickness=2)

                print('%s, Distance: %s' % (dist < 0.6, dist))
            else:
                pass

        if not found:
            cv2.rectangle(img_rgb, rects[i][0], rects[i][1], color = (255, 255, 255), thickness=2)
            cv2.putText(img_rgb, 'Unknown', org=(rects[i][1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                             fontScale=1, color =(255, 255, 255), thickness=2)


            # else:
            #     cv2.rectangle(img_rgb, rects[0][0], rects[0][1], color=(255, 255, 255), thickness=2)
            #     cv2.putText(img_rgb, 'Unknown', org=(rects[0][1][0], rects[0][0][1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #                 fontScale=1, color=(255, 255, 255), thickness=2)


        # if not found:
        #     cv2.rectangle(img_rgb, rects[0][0], rects[0][1], color=(255, 255, 255), thickness=2)
        #     cv2.putText(img_rgb, 'Unknown', org=(rects[0][1][0], rects[0][0][1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #                 fontScale=1, color=(255, 255, 255), thickness=2)





    cv2.imshow('Video', img_rgb)
    if cv2.waitKey(1) == ord('q') :
        break


import cv2
import numpy as np

cap = cv2.VideoCapture(1)
targeted_img = cv2.imread("marvel-final.png")
targeted_video = cv2.VideoCapture("world_video.mp4")


detection = False
frameCounter = 0


success , img_video = targeted_video.read()
hT, wT, cT = targeted_img.shape
img_video = cv2.resize(img_video,(wT,hT))
# print(hT)
# print(wT)

orb =cv2.ORB_create(nfeatures=1000)
kp1 , des1 = orb.detectAndCompute(targeted_img,None)
# targeted_img = cv2.drawKeypoints(targeted_img,kp1,None)


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray [0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                        imgArray [x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray [x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor =[imageBlank]*rows
        hor_con =[imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x  in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray [x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


while True:
    success , img_webcam = cap.read()
    imgAUG = img_webcam.copy()
    kp2, des2 = orb.detectAndCompute(img_webcam, None)
    img_webcam_drawkeypoint = cv2.drawKeypoints(img_webcam, kp2, None)

    if detection == False:
        targeted_video.set(cv2.CAP_PROP_POS_FRAMES,0)
        frameCounter = 0
    else:
        if frameCounter == targeted_video.get(cv2.CAP_PROP_FRAME_COUNT):
            targeted_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
        success, img_video = targeted_video.read()
        img_video = cv2.resize(img_video, (wT, hT))


    brute_force = cv2.BFMatcher()
    matches = brute_force.knnMatch(des1,des2,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75 *n.distance:
            good.append(m)
    # print(len(good))
    imgFeatures = cv2.drawMatches(targeted_img,kp1,img_webcam,kp2,good,None,flags=2)







































































































    if len(good) > 20:
        detection = True
        srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1, 2)
        dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1, 2)
        matrix, mask = cv2.findHomography(srcPts,dstPts,cv2.RANSAC,5)
        # print(matrix)

        pts = np.float32([[0,0],[0,hT],[wT,hT],[wT,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,matrix)
        border_img = cv2.polylines(img_webcam,[np.int32(dst)],True,(255,0,255),3)

        imgWrap = cv2.warpPerspective(img_video,matrix,(img_webcam.shape[1],img_webcam.shape[0]))
        masknew = np.zeros((img_webcam.shape[0],img_webcam.shape[1]),np.uint8)
        cv2.fillPoly(masknew,[np.int32(dst)],(255,255,255))
        maskinverse = cv2.bitwise_not(masknew)
        imgAUG = cv2.bitwise_and(imgAUG,imgAUG,mask= maskinverse)
        imgAUG = cv2.bitwise_or(imgWrap,imgAUG)

        # imgStack = stackImages(0.7,([border_img,targeted_img,img_webcam_drawkeypoint],[imgFeatures,img_video,imgAUG]))
        imgStack = stackImages(0.7,([border_img,targeted_img],[imgFeatures,img_webcam_drawkeypoint]))
        cv2.imshow("ImageStack", imgStack)

        # cv2.imshow("imgwarp", imgWrap )
        # cv2.imshow("img2", border_img )
        # cv2.imshow("features image",imgFeatures)
        # cv2.imshow("targeted image",targeted_img)
        # cv2.imshow("targeted video img",img_video)
        # cv2.imshow("masknew", imgAUG )
        # cv2.imshow("webcam img",img_webcam)
        cv2.waitKey(1)
    frameCounter +=1
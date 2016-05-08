#!/usr/bin/python

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import cv2

def preProcess(img):
    # gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # denoise
    sigColor = 10
    sigSpace = 3
    denoise = cv2.bilateralFilter(gray, -1, sigColor, sigSpace)

    # binarization
    _, binarization = cv2.threshold(denoise, 170, 255, cv2.THRESH_BINARY)# | cv2.THRESH_OTSU)
    #binarization[0:140] = 0
    binarization[0:220] = 0
    #binarization[0:200] = 0


    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    denoise = cv2.cvtColor(denoise, cv2.COLOR_GRAY2BGR)
    binarization = cv2.cvtColor(binarization, cv2.COLOR_GRAY2BGR)
    return (gray, denoise, binarization)

def selectFeatures(img):
    surf = cv2.xfeatures2d.SURF_create(200)
    kp, des = surf.detectAndCompute(img, None)
    filt = []
    for k in kp:
        if(k.size < 35):
            filt.append(k)
    return filt

#def leastSquares(points):
#    x = [p[0] for p in points]
#    y = [p[1] for p in points]
#
#    A = np.vstack([x, np.ones(len(x))]).T
#    m, c = np.linalg.lstsq(A, y)[0]
#    return (m, c)

#def getLines(points):
#    xy = np.array([(p.pt[0], p.pt[1]) for p in points])
#    pred = KMeans(n_clusters=3).fit_predict(xy)
#    classifiedPoints = [[],[],[]]
#    for i in range(len(pred)):
#        km = pred[i]
#        classifiedPoints[km].append(xy[i])
#
#    x1 = 0
#    x2 = 800
#    lines = []
#    for pts in classifiedPoints:
#        m, c = leastSquares(pts)
#        lines.append(((x1, ((x1*m)+c).astype(int)), (x2, ((x2*m)+c).astype(int))))
#
#    return lines

#def perspectiveTransform(img):
#    pts1 = np.float32([[0,0],[0,359],[639,359],[639,0]])
#    pts2 = np.float32([[325,0],[0,359],[639,359],[375,0]])
#    distA = (325, 195)
#    distB = (0, 359)
#    distC = (639, 359)
#    distD = (350, 195)
#    pts2 = np.float32([distA,distB,distC,distD])
#    M = cv2.getPerspectiveTransform(pts1,pts2)
#    M = np.linalg.inv(M)
#
#    #imgDistort = cv2.warpPerspective(img,M,(640,360))
#    #imgDistort = cv2.line(img, distA, distB, (255,255,0),5)
#    #imgDistort = cv2.line(imgDistort, distB, distC, (255,255,0),5)
#    #imgDistort = cv2.line(imgDistort, distC, distD, (255,255,0),5)
#    #imgDistort = cv2.line(imgDistort, distD, distA, (255,255,0),5)
#
#    imgDistort = cv2.warpPerspective(img,M,(640,360))
#    return imgDistort

def getAverage(features):
    size = len(features)
    if size == 0:
        return (0,0)
    avgX = np.average([p.pt[0] for p in features]).astype(int)
    avgY = np.average([p.pt[1] for p in features]).astype(int)
    return (avgX, avgY)

def compare(im1, im2 ,axis):
    return np.concatenate((im1.astype('uint8'), im2.astype('uint8')), axis=axis)

def main():
    #cap = cv2.VideoCapture('vid/road.mp4')
    #cap = cv2.VideoCapture('vid/vtest.mov')
    #cap = cv2.VideoCapture('vid/last.mov')
    cap = cv2.VideoCapture('vid/demov1.mov')
    binaries = []
    rrIndex = 0
    rrMax = 20
    #rrMax = 7

    avgs = []
    rrAvgIndex = 0
    rrAvgMax = 5

    vels = []
    rrVelIndex = 0
    rrVelMax = 10
    averagePt = None
    frameSkip = 300
    frameCount = 0

    drifts = []
    rrDriftIndex = 0
    rrDriftMax = 10
    maxDrift = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
             break;
        frameCount += 1

        gray, denoise, binarization = preProcess(frame)

        # Round robin for or'ing together truncated shit
        rrIndex +=1
        rrIndex = rrIndex%rrMax
        if len(binaries) < rrMax:
            binaries.append(binarization)
        else:
            binaries[rrIndex] = binarization

        ored = None
        for x in binaries:
            if ored is None:
                ored = np.array(x)
            else:
                ored = ored | np.array(x)

        if frameCount > frameSkip:
            features = selectFeatures(ored)
        else:
            features = []
        img_features = cv2.drawKeypoints(frame.copy(),features,None,(255,0,0),4)

        if averagePt is None:
            oldAverage = (0,0)
        else:
            oldAverage = averagePt

        averagePt = getAverage(features)

        # Round robin for average smoothing shit
        #rrAvgIndex += 1
        #rrAvgIndex = rrAvgIndex%rrAvgMax
        #if len(avgs) < rrAvgMax:
        #    avgs.append(averagePt)
        #else:
        #    avgs[rrAvgIndex] = averagePt

        #if len(avgs) > 1:
        #    oldSmooth = smoothAverage
        #else:
        #    oldSmooth = averagePt
        #smoothAverage = tuple(np.average(np.array(avgs).reshape(2,len(avgs)), axis=1).astype(int))

        velocity = np.array(averagePt) - np.array(oldAverage)

        # Round robin for smoothing velocity SHEEEEIIIIT
        rrVelIndex += 1
        rrVelIndex = rrVelIndex%rrVelMax
        if len(vels) < rrVelMax:
            vels.append(velocity)
        else:
            vels[rrVelIndex] = velocity
        smoothVelocity = tuple((np.average(np.array(vels).reshape(2,len(vels)), axis=1)*5).astype(int))
        #smoothVelocity = (smoothVelocity[0]**2, smoothVelocity[1]**2)

        #if len(avgs) > 1:
        #    oldSmooth = smoothAverage
        #else:
        #    oldSmooth = averagePt
        #smoothAverage = tuple(np.average(np.array(avgs).reshape(2,len(avgs)), axis=1).astype(int))

        #img_average = cv2.circle(frame.copy(), smoothAverage, 3, (255,0,255), 2)
        centerOfScreen = np.array((len(frame[0])/2, len(frame)/2)).astype(int)
        adjAvg = (centerOfScreen + smoothVelocity).astype(int)
        img_velocity = cv2.arrowedLine(ored.copy(), tuple(centerOfScreen), (adjAvg[0], centerOfScreen[1]), (0,255,0), 2)
        img_velocity = cv2.arrowedLine(img_velocity, tuple(centerOfScreen), (centerOfScreen[0], adjAvg[1]), (0,0,255), 2)



        xsize = abs(smoothVelocity[0])

        rrDriftIndex += 1
        rrDriftIndex = rrDriftIndex%rrDriftMax
        if len(drifts) < rrDriftMax:
            drifts.append(xsize)
        else:
            drifts[rrDriftIndex] = xsize


        text = "Drifting: {}".format(max(drifts))

        if max(drifts) >= 20:
            img_velocity = cv2.putText(img_velocity, text, (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),5)
        else:
            img_velocity = cv2.putText(img_velocity, text, (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1)

        #show_top = compare(frame, binarization, 1)
        show_top = compare(frame, ored, 1)
        show_bot = compare(img_features, img_velocity, 1)
        #show_bot = compare(img_features, img_features, 1)
        #show_bot = compare(img_features,distort, 1)
        show_this = compare(show_top, show_bot, 0)
        cv2.imshow('frame', show_this)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()

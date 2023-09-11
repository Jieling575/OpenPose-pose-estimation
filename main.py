import numpy
import cv2
import mediapipe as mp
import time
import numpy as np
import matplotlib.pyplot as plt

# Specify the paths for the 2 files

protoFile = "/Users/yangjieling/PycharmProjects/opencvtest/pose_deploy_linevec.prototxt" #COCO


weightsFile = "/Users/yangjieling/PycharmProjects/opencvtest/pose_iter_440000.caffemodel" #COCO
# Read the network into Memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

#points pairs for COCO
POSE_PAIRS = [[1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

# Read image
frame = cv2.imread("33759.jpg")
frameCopy = np.copy(frame)
frameCopy2 = np.copy(frame)
# Specify the input image dimensions
inWidth = 640
inHeight = 457

frameWidth = frame.shape[1]
frameHeight = frame.shape[0]
# Prepare the frame to be fed to the network
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
# Set the prepared object as the input blob of the network
net.setInput(inpBlob)

output = net.forward() #(1,44,86,57)

# print(output)

H = output.shape[2] #86
W = output.shape[3] #57


# temp = output[0,0,:,:]
# minVal0, prob0, minLoc0, point0 = cv2.minMaxLoc(temp)
# print(minVal0, prob0, minLoc0, point0)

#Empty list to store the detected keypoints
points = []
confidence = []
for i in range(18):
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]

    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    # Scale the point to fit on the original image
    x = (frameWidth * point[0]) / W
    y = (frameHeight * point[1]) / H

    #print(i,"point's confidence is:", round(prob,2))
    #print(i,"point's location is:", "(",round(x,2),round(y,2),")")

    if prob > 0.1:
        cv2.circle(frameCopy, (int(x), int(y)), 6, (0, 255, 255),
                   thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frameCopy, "{}".format(i), (int(x), int(
            y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        # Add the point to the list if the probability is greater than the threshold
        points.append((round(x,2), round(y,2)))
        confidence.append(round(prob,2))
    else:
        points.append(None)
        confidence.append(None)

# Draw skeleton for COCO
for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]

    if points[partA] and points[partB]:
        cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
        cv2.circle(frame, points[partA], 6, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

i = 0
'''POSE_CORRECT = [[367, 81],[374, 73],[360, 75],[386, 78],[356, 81],[399, 108],[358, 129],
                [433, 142],[341, 159],[449, 165],[309, 178],[424, 203],[393, 214],
                [429, 294], [367, 273], [466, 362], [396, 341]]
                '''
POSE_CORRECT = [[170, 117], [173, 106], [159, 109], [182, 105], [139, 116],
                [195, 132], [141, 166], [170, 179], [162, 232], [115, 219],
                [118, 232], [227, 236], [179, 251], [239, 343], [155, 355],
                [250, 425], [115, 417]]

'''
for pair in POSE_CORRECT:
    x = pair[0]
    y = pair[1]
    cv2.circle(frameCopy2, (x, y), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
    cv2.putText(frameCopy2, "{}".format(i), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    i += 1
'''
CORRECT_POINTS = POSE_CORRECT.copy()
CORRECT_POINTS.append((0,0))
CORRECT_POINTS[1] = (0,0)
CORRECT_POINTS[2] = POSE_CORRECT[6]
CORRECT_POINTS[3] = POSE_CORRECT[8]
CORRECT_POINTS[4] = POSE_CORRECT[10]
CORRECT_POINTS[6] = POSE_CORRECT[7]
CORRECT_POINTS[7] = POSE_CORRECT[9]
CORRECT_POINTS[8] = POSE_CORRECT[12]
CORRECT_POINTS[9] = POSE_CORRECT[14]
CORRECT_POINTS[10] = POSE_CORRECT[16]
CORRECT_POINTS[12] = POSE_CORRECT[13]
CORRECT_POINTS[13] = POSE_CORRECT[15]
CORRECT_POINTS[14] = POSE_CORRECT[2]
CORRECT_POINTS[15] = POSE_CORRECT[1]
CORRECT_POINTS[16] = POSE_CORRECT[4]
CORRECT_POINTS[17] = POSE_CORRECT[3]

for pair in CORRECT_POINTS:
    x = pair[0]
    y = pair[1]
    cv2.circle(frameCopy2, (x, y), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
    cv2.putText(frameCopy2, "{}".format(i), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    i += 1

#print(points)
#print(CORRECT_POINTS)

points.pop(1)
CORRECT_POINTS.pop(1)
confidence.pop(1)
error = []

for i in range(len(points)):
    x_e = np.abs(points[i][0] - CORRECT_POINTS[i][0])
    y_e = np.abs(points[i][1] - CORRECT_POINTS[i][1])
    e = np.sqrt(x_e**2 + y_e **2)
    error.append(round(e,2))

print(error)
errorSmall = error.copy()
for i in range(len(error)):
    errorSmall[i] = round(error[i]/15,2)

print(errorSmall)

x = [i for i in range(17)]
plt.plot(x, confidence, label="confidence",marker="o")
plt.plot(x, errorSmall, label="error",marker="o")

plt.xlabel('key points')
plt.legend()
plt.show()

cv2.imshow("Output-Keypoints", frameCopy)
cv2.imshow('Output-Skeleton', frame)
cv2.imshow("correct", frameCopy2)

cv2.waitKey(0)
cv2.destroyAllWindows()

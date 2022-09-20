import numpy as np
import cv2 as cv
from time import time

start = time()

kernel1 = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

kernel2 = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

kernel3 = np.array([
    [-1, -2, 0, -2, -1],
    [-2, -3, 0, -3, -2],
    [0, 0, 0, 0, 0],
    [2, 3, 0, 3, 2],
    [1, 2, 0, 2, 1]
])


img = cv.imread('benchMarkImage.png', cv.IMREAD_GRAYSCALE)

output = np.zeros(np.shape(img))

def conv2D(input, kernel):
    output = np.zeros(np.shape(input))
    for row in range(1,np.shape(input)[0]-np.shape(kernel)[0]+1):
        for column in range(1, np.shape(input)[1]-np.shape(kernel)[0]+1):
            endRow = row + np.shape(kernel)[0]
            endCol = column+np.shape(kernel)[1]
            output[row:endRow,column:endCol] = np.add(output[row:endRow,column:endCol],np.multiply(input[row,column],kernel))

    
    #output = np.add(np.abs(output.min()), output)
    #outputMap = output.max()/256
    output = np.abs(output)
    
    output = np.divide(output, output.max())
    return(output)

result1 = conv2D(img, kernel1)


result2 = conv2D(img, kernel2)

result3 = conv2D(img, kernel3)
result = np.divide(np.add(result1, result2),2)


test = conv2D(result, kernel3)



end = time()

print(end-start)

cv.imshow("image", result)

cv.waitKey(0)

cv.destroyAllWindows()
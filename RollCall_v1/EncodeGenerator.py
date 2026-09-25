''' This file detects faces on image files and generates encoding of the faces
    Author: Sangmork park, Virginia Military Institute
    Version: Mar. 2024
    Source file: *.png
    Output file: EncodFile.p
'''

import cv2
import face_recognition
import pickle
import os

''' generate the encode list of image files '''
def getEncodings(imgList):
    encodeList = [];
    for img in imgList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    # print(encodeList)
    return encodeList

''' main() method'''
def generateEncodes():

    ''' importing student images'''
    folderPath = 'Images/cis112'
    imgFileList = os.listdir(folderPath)
    # print(imgFileList)

    imgList = []
    studentIDs = []
    for fileName in imgFileList:
        imgList.append(cv2.imread(os.path.join(folderPath, fileName)));
        studentIDs.append(os.path.splitext(fileName)[0])
        print(fileName)
        # print(os.path.splitext(fileName)[0])
    # print(studentIDs)

    print("Encoding Started.")
    encodeListKnown = getEncodings(imgList)
    encodeListKnowWithIDs = [encodeListKnown, studentIDs]
    # print(encodeListKnowWithIDs)
    print("Encoding Completed.")

    ''' save encoding data into a file'''
    outFile = open("EncodeFile.cis112", 'wb')
    pickle.dump(encodeListKnowWithIDs, outFile)
    outFile.close()
    print("Encoding File Saved.")

if __name__ == '__main__':
    generateEncodes()



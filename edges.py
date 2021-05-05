import cv2
cap = cv2.VideoCapture(0) # 0 local camera
con = 0
while cap.isOpened():
  
    #BGR image feed from camera
  ret, img = cap.read()
  #BGR to grayscale
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(img_gray,10,20,apertureSize = 3)
  
  
  cv2.imshow("Edges", edges)
  
  k = cv2.waitKey(10)
  if k==27:
    break

cap.release()
cv2.destroyAllWindows()

import cv2

cap=cv2.VideoCapture("eye_motion.mp4")

while 1:
    ret,frame=cap.read()
    if ret is False:
        break

    roi=frame[80:210,230:450]#videodaki göz bulunan bölgeyi deneme yanılma ile tespit ettik ve roi olarak atadık
    rows,cols,_=roi.shape

    gray=cv2.cvtColor(roi,cv2.COLOR_BGRA2GRAY)
    _,threshold=cv2.threshold(gray,3,255,cv2.THRESH_BINARY_INV)#görüntüyü binary formata (yani 0-1 siyah-beyaz) çevirdik

    contours,_=cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours=sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)#contour alanlarını büyüğe göre sıraladık

    for cnt in contours:
        (x,y,w,h)=cv2.boundingRect(cnt)#w genişlik h yükseklik... ilgilenilen bölgeyi vurgulamak için boundingrect kullanılır.
        cv2.rectangle(roi,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.line(roi,(x+int(w/2) ,0),(x+int(w/2),rows),(0,255,0),2)
        cv2.line(roi, (0,y+int(h/2)), (cols, y+int(h/2)), (0, 255, 0), 2)
        break
    frame[80:210, 230:450]=roi
    cv2.imshow("frame",frame)


    if cv2.waitKey(80) & 0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
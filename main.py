import cv2
import time
import random
import mediapipe as mp
import numpy as np
from numpy.lib.index_tricks import ix_

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def draw_text(img, text,  
          pos=(0, 0),
          font=cv2.FONT_HERSHEY_COMPLEX,
          font_scale=.8,
          font_thickness=1,
          text_color=(255, 255, 255),
          text_color_bg=(0, 255, 0, 10)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + (text_w * 2), y - (text_h * 2)), text_color_bg, -1)
    cv2.putText(img, text, (x+(text_w >> 1), y - (text_h>>1)), font, font_scale, text_color, font_thickness)

    return text_size

def music(tones):
    import winsound
    winsound.PlaySound(tones, winsound.SND_FILENAME|winsound.SND_NOWAIT)

def beep(name):
    import threading
    x = threading.Thread(target=music, args=(name,), daemon=True)
    x.start()

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask=None):
    """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.

    `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
    """
    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]

    if alpha_mask:
        alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
        alpha_inv = 1.0 - alpha

    img_crop[:] = img_overlay_crop

def getAngle(vx, vy):
    if vx > 0:
        if vy < 0: return 315
        return 225
    else:
        if vy < 0: return 45
        return 135        
       
if __name__ == "__main__":

    window = 'GAMEFia'

    cv2.namedWindow(window, cv2.WND_PROP_FULLSCREEN)
    #cv2.setWindowProperty(window,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    # For webcam input:
    cap = cv2.VideoCapture(0)
    play = cv2.VideoCapture("barata.gif")

    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.5) as hands:


        # Initializing current time and precious time for calculating the FPS
        previousTime = 0
        currentTime = 0
        previousDistance = -1
        pressed = False
        ptimed = 0
        holded = False
        tried = False
        lx = cx = 200
        lx = cy = 200
        vx = 5
        vy = 5
        a = 225
        score = 0
        level = 5
        ix = iy = 0

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            h, w, d = image.shape  

            # Draw the hand annotations on the image.
            blank_image = np.full((h,w,d), 255, np.uint8)
            blank_image.flags.writeable = True
            blank_image = cv2.cvtColor(blank_image, cv2.COLOR_RGB2BGR)
            
            ok, barata = play.read()

            bsize = 100
            bmargin = bsize // 5
   
            if ok:
           
                barata = cv2.resize(barata, (bsize, bsize))
                M = cv2.getRotationMatrix2D((bsize//2, bsize//2), a if not holded else a + random.randrange(30), 1.0)
                barata = cv2.warpAffine(barata, M, (bsize, bsize), borderValue=(255,255,255))      
                overlay_image_alpha(blank_image, barata, cx-bmargin, cy-bmargin)
                #cv2.rectangle(blank_image, (cx, cy-bmargin), (cx+bsize-bmargin*2, cy+bsize-bmargin), (255, 0, 0), 2)
    
            else: 
                play.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, barata = play.read()
                continue
                #barata = cv2.resize(barata, (bsize, bsize))
                #overlay_image_alpha(blank_image, barata, -bmargin, -bmargin)

                # blank_image = cv2.circle(blank_image, (cx,cy), 10, (0,0,255),-1)

        
            # resizing the frame for better view
            # image = cv2.resize(image, (800, 600))
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Flip the image horizontally for a selfie-view display.
            image = cv2.flip(image, 1)
            
            results = hands.process(image)

 
            lx = cx
            ly = cy

            if not holded:
                cx += vx
                cy += vy   
                if cx+bsize-bmargin*2 > w or cx < 0: 
                    vx = -vx
                    a = getAngle(vx,vy)
                if cy+bsize-bmargin > h or cy < 0: 
                    vy = -vy
                    a = getAngle(vx,vy)
            else:
                play.set(cv2.CAP_PROP_POS_FRAMES, 0)

              
            # if results.multi_handedness:
               # for multi_handedness in results.multi_handedness:

            if results.multi_hand_landmarks:
                # for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(results.multi_hand_landmarks)):
                    hand_landmarks = results.multi_hand_landmarks[i]
                    handedness = results.multi_handedness[i]
                    allmarks = [hand_landmarks.landmark[i] for i in [0, 1, 5, 17]]
                   
                    x_max = int(max(allmarks, key = lambda item: item.x).x * w)
                    y_max = int(max(allmarks, key = lambda item: item.y).y * h)
                    x_min = int(min(allmarks, key = lambda item: item.x).x * w)
                    y_min = int(min(allmarks, key = lambda item: item.y).y * h)
             
                    wrist = hand_landmarks.landmark[0]
                    mcp = hand_landmarks.landmark[5]
                    dst = ((wrist.x-mcp.x) ** 2 + (wrist.y-mcp.y)**2)
                    dst = 11.2 / (dst ** .5)

                    if previousDistance > 0 and abs(previousDistance - dst) > 4:
                        if not pressed and previousDistance > dst:
                            pressed = True
                            ix = (x_max + x_min) // 2
                            iy = (y_max + y_min) // 2
                            ptimed =  time.time()
 
                            
                    if  time.time()-ptimed > 3:
                        pressed = False
                        holded = False  
                        tried = False
                        previousDistance = dst 
                        ptimed =  time.time()      
 
                   
                    previousDistance = dst 

                    mp_drawing.draw_landmarks(
                        blank_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())


                    if pressed and not tried:
                        tried = True
                        borderx = cx+((bsize-bmargin * 2) >> 1)
                        bordery = cy-bmargin+((bsize-bmargin) >> 1)   

                        #if pressed:
                            #cv2.circle(blank_image, (borderx, bordery), 10, (255, 0, 0))
                            #cv2.rectangle(blank_image, (cx, cy-bmargin), (cx+bsize-bmargin*2, cy+bsize-bmargin), (255, 0, 0), 2)

                        if not holded and borderx > x_min and borderx < x_max and bordery > y_min and bordery < y_max:
                            holded = True
                            score += 1   
                            beep("angry.wav")
                            vx = vx if random.randrange(30) < 15 else -vx
                            vy = vy if random.randrange(30) < 15 else -vy
                            a = getAngle(vx,vy)
                            if level - score == 0:
                                vx += vx//5
                                vy += vy//5
                                if abs(vx) < 15:
                                    level = 10 + score//2


                        else:
                            beep("metal.wav")
                              
            
                    if pressed:    
                        cv2.circle(blank_image, (ix, iy), bmargin,  (80, 80, 80), -1)
                    #draw_text(blank_image, f"{handedness.classification[0].label} {dst:.0f}cm", (x_min, y_min))


            # Calculating the FPS
            currentTime = time.time()
            fps = int(1 / (currentTime-previousTime))
            previousTime = currentTime
            
            
            # Displaying FPS on the image
            # cv2.putText(blank_image, f"{fps} FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.putText(blank_image, f"Hits: {score}", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow(window, blank_image)
            # cv2.imshow('MediaPipe Hands', image)

            if cv2.waitKey(5) & 0xFF == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_FULLSCREEN) < 0:
                break

    cap.release()
    cv2.destroyAllWindows()
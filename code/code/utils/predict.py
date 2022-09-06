from google.colab.patches import cv2_imshow
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array


def predicted(model, path_haar, path_img):

    classifier =cv2.CascadeClassifier(cv2.data.haarcascades + path_haar)

    img = cv2.imread(path_img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_detected = classifier.detectMultiScale(gray_img, 1.18, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255.0

        predictions = model.predict(img_pixels)
        max_index = int(np.argmax(predictions))

        emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
        predicted_emotion = emotions[max_index]


        cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    resized_img = cv2.resize(img, (256, 256))
    cv2_imshow( resized_img)



def get_imgs_test(test_ds):
  test = list(test_ds.as_numpy_iterator())
  labels = []
  images = []   # images to test

  for i in range(len(test)):
      labels.extend(np.array(test[i][1]))
      images.extend(np.array(test[i][0]))
      
  images = np.array(images)
  return images, labels

def plot_wrong_image(images, test_labels, emotion):
    Y_pred = model.predict(images)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = np.argmax(test_labels, axis=1)

    wrong_images, T_labels, F_labels = [], [] ,[]
    for i, test_image in enumerate(images):
        if y_pred[i] == y_true[i]:
            pass
        else:
            wrong_images.append(test_image)
            T_labels.append(y_true[i])
            F_labels.append(y_pred[i])
    
    
    plt.figure(figsize=(25,20))
    for i in range(5):
        idx = np.random.randint(0, len(wrong_images) - 1)
        plt.subplot(1, 5, i+1)
        img = wrong_images[idx]
        plt.imshow(img[:,:,0], cmap='gray')

        True_labels =  emotion[T_labels[idx]]
        False_labels = emotion[F_labels[idx]]

        plt.title('Actual class:{} , Predicted class:{}'.format(True_labels, False_labels), size=12, color="black") 
        plt.xticks([])
        plt.yticks([])
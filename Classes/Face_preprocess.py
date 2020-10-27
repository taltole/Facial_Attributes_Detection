import cv2
import hashlib, os
import numpy as np


class BasicFunction:

    def __init__(self, folderpath):
        """
        :param folderpath: path to the image folder
        """
        self.folderpath = folderpath

    def remove_dup(self):
        """
        Remove duplicates images in a folder
        """
        duplicates = []
        hash_keys = dict()

        os.chdir(self.folderpath)
        file_list = os.listdir()
        print('number of images before removing duplicates', len(file_list))

        for index, filename in enumerate(os.listdir('.')):  # listdir('.') = current directory
            if os.path.isfile(filename):
                with open(filename, 'rb') as f:
                    filehash = hashlib.md5(f.read()).hexdigest()
                if filehash not in hash_keys:
                    hash_keys[filehash] = index
                else:
                    duplicates.append((index, hash_keys[filehash]))

        print('number of duplicates', len(duplicates))
        print('number of images after removing', len(file_list) - len(duplicates))

        for index in duplicates:
            os.remove(file_list[index[0]])

        print('Remove duplicates images from', self.folderpath)

    def crop_face(self, new_foldername, image_name):
        """
            This function detects faces on images and cropped them. Images are saved in a new folder.
        :param new_foldername: path to the folder with the cropped faces
        :param image_name: name of the new images
        """
        file_types = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG')

        files = [file_i for file_i in os.listdir(self.folderpath) if file_i.endswith(file_types)]

        filenames = [os.path.join(self.folderpath, fname)
                     for fname in files]

        count = 0
        image_number = 0
        for file in filenames:
            image_number += 1
            print(' image number ', image_number)
            image = cv2.imread(file)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))

            print("[INFO] Found {0} Faces.".format(len(faces)))

            for (x, y, w, h) in faces:
                count += 1
                w = w + 50
                h = h + 150
                p = 50
                crop_img = image[y - p + 1:y + h + p, x + 1:x + w]

                print("[INFO] Object found. Saving locally.")
                try:
                    sharpen = cv2.resize(crop_img, (150, 150), interpolation=cv2.INTER_AREA)  # try something else

                    if not os.path.exists(new_foldername):
                        os.makedirs(new_foldername)
                    cv2.imwrite(new_foldername + "/" + image_name + '_' + str(count) + ".jpg", sharpen)
                except:
                    pass
        print('Images saved in', new_foldername)

    def preprocessing(self, img_size):
        """
        This function is doing general preprocessing on images
        :param img_size: size of the image
        :return: list of preprocessed images
        """
        images_list = os.listdir(self.folderpath)
        preprocess_img = []

        # Images Preprocess
        IMG_WIDTH = img_size
        IMG_HEIGHT = img_size

        for img in images_list:
            try:
                image = cv2.imread(os.path.join(self.folderpath, img))
                image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
                image = cv2.cvtColor(image, cv2.cv2.CAP_OPENNI_GRAY_IMAGE)
                image = np.array(image).astype('float32') / 255.
                preprocess_img.append(image)
            except:
                print('ERROR', img)
        return preprocess_img

    @staticmethod
    def print_summary(model):
        """
        This function returns a model summary
        :param model: model
        :return: summary
        """
        print(f"Input_shape:\t{model.input_shape}\nOutput_shape:\t{model.output_shape}\nParams:\t{model.count_params()}"
              f"\nLayers:\t{len(model.layers)}\n\n")
        return model.summary()

    # def save_model(model):

from config import *


class DataAugmentation:
    """
    This Class contains functions related to data augmentation.
    """
    def __init__(self, out_directory, legend_out, img_directory, images_list):
        """
        legend_out: string
        img_directory : path to the image folder
        out_directory: path to the folder where the new images are saved
        images_list: path to the csv file
        """
        self.legend_out = legend_out
        self.img_directory = input('Enter path for images folder: ')
        self.out_directory = os.path.join(img_directory, 'augmented')
        self.images_list = images_list

        # Set the image directories. These can be overridden in the function calls.
        # Creating the output file legend.csv
        with open(out_directory + 'legend.csv', 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(['image', 'label'])


    def save_image(self, img, name, label):
        """
        Function for saving the file and writing the legend file.
        """
        with open(self.legend_out + 'new_files_list.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([img, name, label])
        cv2.imwrite(self.out_directory + name, img)

    def hflip_img(self, image, label):
        """
        Performs the horizontal flip of the image.
        """
        # Load raw image file into memor
        img = cv2.imread(self.img_directory + image)
        res = cv2.flip(img, 1)  # Flip the image
        self.save_image(res, 'hflip' + image, label)

    def vflip_img(self, image, label):
        """
        Performs a vertical flip of the image.
        """
        # Load raw image file into memor
        img = cv2.imread(self.img_directory + image)
        res = cv2.flip(img, 0)  # Flip the image
        self.save_image(res, 'vflip' + image, label)

    def rotate_img(self, image, label, angle):
        """
        Rotates the image given a specific number of degrees, positive is clockwise negative is counterclockwise.
        """
        img = cv2.imread(self.img_directory + image, 0)
        rows, cols = img.shape

        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        self.save_image(dst, str(angle) + 'rotate' + image, label)

    def shift_img(self, image, label, x, y):
        """
        Translates the image horizontally and vertically, postive is down and right negative is up and left.
        """
        img = cv2.imread(self.img_directory + image, 0)
        rows, cols = img.shape

        M = np.float32([[1, 0, x], [0, 1, y]])
        dst = cv2.warpAffine(img, M, (cols, rows))
        self.save_image(dst, str(x) + '_' + str(y) + 'shift' + image, label)

    def blur_img(self, image, label, size=5):
        """
        Blurs the image using the average value from the 5X5 pixle square surrounding each pixel.
        """
        img = cv2.imread(self.img_directory + image, 0)
        blur = cv2.blur(img, (size, size))
        self.save_image(blur, 'blur' + image, label)

    def gauss_img(self, image, label, size=5):
        """
        Blurs the image using Gaussian weights from the 5X5 pixle square surrounding
        each pixel.
        """
        img = cv2.imread(self.img_directory + image, 0)
        blur = cv2.GaussianBlur(img, (size, size), 0)
        self.save_image(blur, 'gauss' + image, label)

    def bilateral_img(self, image, label, size=5):
        """
        Applys a bilateral filter that sharpens the edges while bluring the other areas.
        """
        img = cv2.imread(self.img_directory + image, 0)
        blur = cv2.bilateralFilter(img, 9, 75, 75)
        self.save_image(blur, 'bilat' + image, label)

    def main(self):
        # Loading the basic legend file before augmentation
        legend = pd.read_csv(self.images_list)
        files = legend['image']
        labels = legend['label']
        i = 0

        # Running the augmentations
        for f in files:
            label = labels[i]
            self.hflip_img(f, label)
            self.vflip_img(f, label)  # Dont use on faces image
            self.rotate_img(f, label, 15)
            self.rotate_img(f, label, -15)
            self.rotate_img(f, label, 30)
            self.rotate_img(f, label, -30)
            self.shift_img(f, label, 50, 50)
            self.shift_img(f, label, -50, -50)
            self.blur_img(f, label)
            self.gauss_img(f, label)
            self.bilateral_img(f, label)
            i += 1

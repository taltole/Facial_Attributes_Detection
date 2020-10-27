from config import *


class DataAugmentation:
    def __init__(self, out_directory, legend_out, img_directory):
        self.legend_out = legend_out
        self.img_directory = input('Enter path for images folder: ')
        self.out_directory = os.path.join(img_directory, 'augmented')
        self.legend_out = 'data/augmented/'
        self.images_list = 'files list.csv'

        # Set the image directories. These can be overridden in the function calls.
        # Creating the output file legend.csv
        with open(out_directory + 'legend.csv', 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(['image', 'label'])

    # Function for saving the file and writing the legend file.
    def save_image(self, img, name, label, out_folder=self.out_directory, legend=self.legend_out):
        with open(legend + 'new_files_list.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([img, name, label])
        cv2.imwrite(out_folder + name, img)

    # Performs the horizontal flip of the image.
    def hflip_img(self, image, label, in_folder=self.img_directory):
        # Load raw image file into memor
        img = cv2.imread(in_folder + image)
        res = cv2.flip(img, 1)  # Flip the image
        self.save_image(res, 'hflip' + image, label)

    # Performs a vertical flip of the image.
    def vflip_img(self, image, label, in_folder=self.img_directory):
        # Load raw image file into memor
        img = cv2.imread(in_folder + image)
        res = cv2.flip(img, 0)  # Flip the image
        self.save_image(res, 'vflip' + image, label)

    # Rotates the image given a specific number of degrees, positive is clockwise negative is counterclockwise.
    def rotate_img(self, image, label, angle, in_folder=img_directory):
        img = cv2.imread(in_folder + image, 0)
        rows, cols = img.shape

        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        self.save_image(dst, str(angle) + 'rotate' + image, label)

    # Translates the image horizontally and vertically, postive is down and right negative is up and left.
    def shift_img(self, image, label, x, y, in_folder=img_directory):
        img = cv2.imread(in_folder + image, 0)
        rows, cols = img.shape

        M = np.float32([[1, 0, x], [0, 1, y]])
        dst = cv2.warpAffine(img, M, (cols, rows))
        save_image(dst, str(x) + '_' + str(y) + 'shift' + image, label)

    # Blurs the image using the average value from the 5X5 pixle square surrounding each pixel.
    def blur_img(self, image, label, size=5, in_folder=img_directory):
        img = cv2.imread(in_folder + image, 0)
        blur = cv2.blur(img, (size, size))
        save_image(blur, 'blur' + image, label)

    # Blurs the image using Gaussian weights from the 5X5 pixle square surrounding
    # each pixel.
    def gauss_img(self, image, label, size=5, in_folder=img_directory):
        img = cv2.imread(in_folder + image, 0)
        blur = cv2.GaussianBlur(img, (size, size), 0)
        save_image(blur, 'gauss' + image, label)

    # Applys a bilateral filter that sharpens the edges while bluring the other areas.
    def bilateral_img(self, image, label, size=5, in_folder=img_directory):
        img = cv2.imread(in_folder + image, 0)
        blur = cv2.bilateralFilter(img, 9, 75, 75)
        save_image(blur, 'bilat' + image, label)

    def main(self, ):
        # Loading the basic legend file before augmentation
        legend = pd.read_csv(images_list)
        files = legend['image']
        labels = legend['label']
        i = 0

        # Running the augmentations
        for f in files:
            label = labels[i]
            hflip_img(f, label)
            vflip_img(f, label)  # Dont use on faces image
            rotate_img(f, label, 15)
            rotate_img(f, label, -15)
            rotate_img(f, label, 30)
            rotate_img(f, label, -30)
            shift_img(f, label, 50, 50)
            shift_img(f, label, -50, -50)
            blur_img(f, label)
            gauss_img(f, label)
            bilateral_img(f, label)
            i += 1

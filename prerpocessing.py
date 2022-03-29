from PIL import Image
import os
import cv2
import numpy
import colorgram

allImageFiles = os.listdir("AnimeDataset/data/train/")
counter = 0

for namae_saiki in allImageFiles:
    # open image
    img = Image.open("AnimeDataset/data/train/" + namae_saiki).convert('RGB')

    # Crop the image using crop() method
    ColorImage = img.crop((0, 0, 512, 512))
    SketchImage = img.crop((512, 0, 1024, 512))

    # saving colored and sketch images
    ColorImage.save("RealData/Color/" + namae_saiki)
    SketchImage.save("RealData/Sketch/" + namae_saiki)

    # read the seperated images
    color_image = cv2.imread("RealData/Color/" + namae_saiki)
    sketch_image = cv2.imread("RealData/Sketch/" + namae_saiki)

    # seperate it into 4 different images
    pojok_kiri_atas = color_image[0:256, 0:256]
    pojok_kiri_bawah = color_image[256:512, 0:256]
    pojok_kanan_atas = color_image[0:256, 256:512]
    pojok_kanan_bawah = color_image[256:512, 256:512]

    # extract all colors from colored image
    colors = colorgram.extract("RealData/Color/" + namae_saiki, 10)

    # now we need to mark all colors in the color image into sketch image
    # Iki seng buat pojok kiri atas
    for colors_saiki in colors:
        # print(colors_saiki[0])
        red = colors_saiki.rgb.r
        green = colors_saiki.rgb.g
        blue = colors_saiki.rgb.b

        temp_color_check = [red, green, blue]

        indices = numpy.where(pojok_kiri_atas == temp_color_check)
        coordinates = zip(indices[0], indices[1])
        unique_coordinates = list(set(list(coordinates)))

        panjange = len(unique_coordinates) / 2

        if panjange != 0:
            # print(unique_coordinates[int(panjange)])
            this_is_the_spot_x = unique_coordinates[int(panjange)][0]
            this_is_the_spot_y = unique_coordinates[int(panjange)][1]
            # print(this_is_the_spot_x)
            # print(this_is_the_spot_y)
            sketch_image = cv2.circle(sketch_image, (this_is_the_spot_y, this_is_the_spot_x), radius=1,
                                      color=(blue, green, red), thickness=-1)

    # iki seng buat pojok kanan atas
    for colors_saiki in colors:
        # print(colors_saiki[0])
        red = colors_saiki.rgb.r
        green = colors_saiki.rgb.g
        blue = colors_saiki.rgb.b

        temp_color_check = [red, green, blue]

        indices = numpy.where(pojok_kanan_atas == temp_color_check)
        coordinates = zip(indices[0], indices[1])
        unique_coordinates = list(set(list(coordinates)))

        panjange = len(unique_coordinates) / 2

        if panjange != 0:
            # print(unique_coordinates[int(panjange)])
            this_is_the_spot_x = unique_coordinates[int(panjange)][0]
            this_is_the_spot_y = unique_coordinates[int(panjange)][1]
            this_is_the_spot_y += 256
            # print(this_is_the_spot_x)
            # print(this_is_the_spot_y)
            sketch_image = cv2.circle(sketch_image, (this_is_the_spot_y, this_is_the_spot_x), radius=1,
                                      color=(blue, green, red), thickness=-1)

    # iki seng buat kiri bawah
    for colors_saiki in colors:
        # print(colors_saiki[0])
        red = colors_saiki.rgb.r
        green = colors_saiki.rgb.g
        blue = colors_saiki.rgb.b

        temp_color_check = [red, green, blue]

        indices = numpy.where(pojok_kiri_bawah == temp_color_check)
        coordinates = zip(indices[0], indices[1])
        unique_coordinates = list(set(list(coordinates)))

        panjange = len(unique_coordinates) / 2

        if panjange != 0:
            # print(unique_coordinates[int(panjange)])
            this_is_the_spot_x = unique_coordinates[int(panjange)][0]
            this_is_the_spot_y = unique_coordinates[int(panjange)][1]
            this_is_the_spot_x += 256
            # print(this_is_the_spot_x)
            # print(this_is_the_spot_y)
            sketch_image = cv2.circle(sketch_image, (this_is_the_spot_y, this_is_the_spot_x), radius=1,
                                      color=(blue, green, red), thickness=-1)

    # iki seng buat kanan bawah
    for colors_saiki in colors:
        # print(colors_saiki[0])
        red = colors_saiki.rgb.r
        green = colors_saiki.rgb.g
        blue = colors_saiki.rgb.b

        temp_color_check = [red, green, blue]

        indices = numpy.where(pojok_kanan_bawah == temp_color_check)
        coordinates = zip(indices[0], indices[1])
        unique_coordinates = list(set(list(coordinates)))

        panjange = len(unique_coordinates) / 2

        if panjange != 0:
            # print(unique_coordinates[int(panjange)])
            this_is_the_spot_x = unique_coordinates[int(panjange)][0]
            this_is_the_spot_y = unique_coordinates[int(panjange)][1]
            this_is_the_spot_x += 256
            this_is_the_spot_y += 256
            # print(this_is_the_spot_x)
            # print(this_is_the_spot_y)
            sketch_image = cv2.circle(sketch_image, (this_is_the_spot_y, this_is_the_spot_x), radius=1,
                                      color=(blue, green, red), thickness=-1)

    # save image
    status = cv2.imwrite("RealData/Sketch/" + namae_saiki, sketch_image)
    # cv2.imshow("Hasil", sketch_image)
    # cv2.waitKey()
    counter = counter + 1
    print("Image written to file-system : ", status)
    print("Images finished ", counter)

import tensorflow as tf
import matplotlib.pyplot as plt

image_raw_data_jpg = tf.read_file('./images/001.JPG')
image = tf.image.decode_jpeg(image_raw_data_jpg, channels=3)
# image = tf.image.resize_images(image, size=[32, 32])
image = tf.image.convert_image_dtype(image, tf.float32)
bounding_crop = tf.image.crop_to_bounding_box(image, offset_height=200, offset_width=200, target_height=411,
                                              target_width=733)
with tf.Session() as sess:
    img, bounding_crop = sess.run([image, bounding_crop])

    # print(img.shape)
    print(bounding_crop.shape)
    # plt.figure(1)  # 图像显示
    # plt.imshow(img)
    plt.figure(2)  # 图像显示
    plt.imshow(bounding_crop)
    plt.show()


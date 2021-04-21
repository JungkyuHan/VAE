import tensorflow as tf
from mnist_data import MNISTData
from VAE import VAE
import os


def sample_from_gaussian(in_mean, in_sd):
    batch_size = tf.shape(in_mean)[0]
    dim = tf.shape(in_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch_size, dim))
    sample = tf.math.multiply(in_sd, epsilon)
    sample = tf.math.add(in_mean, sample)
    return sample


if __name__ == "__main__":
    print("Hi. I am a VAE generator Tester.")
    latent_vec_dim = 40
    num_epochs = 5
    complexity_loss_weight = 1.0
    reconstrcution_loss_weight = 1.0

    width = 28
    height = 28
    in_out_dim = width * height

    auto_encoder = VAE(input_output_dim=in_out_dim,
                       encoder_common_hidden_layers=[200],
                       encoder_separate_hidden_layers=[100],
                       latent_vec_dim=latent_vec_dim,
                       decoder_hidden_layers=[200, 200],
                       complexity_loss_weight=complexity_loss_weight,
                       reconstrcution_loss_weight=reconstrcution_loss_weight,
                       batch_size=256)

    load_path = "../model/vae_model.ckpt"
    print("load model weights from %s" % load_path)
    auto_encoder.vae.load_weights(load_path)

    f = open("../model/latent_vec_info.txt", "r")
    test_label = []
    latent_vec_mean_per_label = []
    latent_vec_sd_per_label = []
    for i in range(0, 10):
        label_and_dim = f.readline().strip("\n")
        label_and_dim = label_and_dim.split(',')
        label_and_dim = list(map(int, label_and_dim))
        test_label.append(label_and_dim[0])
        # print(label_and_dim)

        latent_vec_mean = f.readline().strip("\n")
        latent_vec_mean = latent_vec_mean.split(',')
        latent_vec_mean = list(map(float, latent_vec_mean))
        mean = tf.constant(latent_vec_mean)
        latent_vec_mean_per_label.append(mean)
        # print(mean)

        latent_vec_sd = f.readline().strip("\n")
        latent_vec_sd = latent_vec_sd.split(',')
        latent_vec_sd = list(map(float, latent_vec_sd))
        sd = tf.constant(latent_vec_sd)
        latent_vec_sd_per_label.append(sd)
        # print(sd)
    f.close()

    latent_vec_mean_per_label = tf.convert_to_tensor(latent_vec_mean_per_label, dtype=tf.float32)
    latent_vec_sd_per_label = tf.convert_to_tensor(latent_vec_sd_per_label, dtype=tf.float32)
    reconst_data_by_mean = auto_encoder.decoder.predict(latent_vec_mean_per_label)

    reconst_data_x_by_mean_print = reconst_data_by_mean.reshape(10, width, height)
    reconst_data_x_by_mean_print = tf.math.sigmoid(reconst_data_x_by_mean_print)
    MNISTData.print_10_images(reconst_data_x_by_mean_print, test_label)

    print("Add noise..")

    for i in range(1, 10):
        sample_latent_vecs = sample_from_gaussian(latent_vec_mean_per_label, latent_vec_sd_per_label)
        reconst_data_by_mean = auto_encoder.decoder.predict(sample_latent_vecs)
        reconst_data_x_by_mean_print = reconst_data_by_mean.reshape(10, width, height)
        reconst_data_x_by_mean_print = tf.math.sigmoid(reconst_data_x_by_mean_print)
        print("%d-th trial" % i)
        MNISTData.print_10_images(reconst_data_x_by_mean_print, test_label)


    print("Bye~")

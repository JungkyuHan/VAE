import tensorflow as tf
from mnist_data import MNISTData
from VAE import VAE
import os


if __name__ == "__main__":
    print("Hi. I am a VAE Tester.")
    latent_vec_dim = 40
    num_epochs = 5
    complexity_loss_weight = 1.0
    reconstrcution_loss_weight = 1.0

    data_loader = MNISTData()
    data_loader.load_data()
    auto_encoder = VAE(input_output_dim=data_loader.in_out_dim,
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

    x_train = data_loader.x_train_dataset
    x_tune = data_loader.x_tune_dataset

    # print for test
    num_test_items = 56
    test_data = data_loader.x_test[0:num_test_items]
    test_label = data_loader.y_test[0:num_test_items]
    test_data = test_data.reshape(num_test_items, data_loader.in_out_dim)
    test_data_x_print = test_data.reshape(num_test_items, data_loader.width, data_loader.height)

    print("const by sample")
    reconst_data = auto_encoder.vae.predict(test_data)
    reconst_data_x_print = reconst_data.reshape(num_test_items, data_loader.width, data_loader.height)
    reconst_data_x_print = tf.math.sigmoid(reconst_data_x_print)
    MNISTData.print_56_pair_images(test_data_x_print, reconst_data_x_print, test_label)

    print("const by mean")
    latent_mean, latent_sd, sampled_latent_vec = auto_encoder.encoder.predict(test_data)
    reconst_data_by_mean = auto_encoder.decoder.predict(latent_mean)
    reconst_data_x_by_mean_print = reconst_data_by_mean.reshape(num_test_items, data_loader.width, data_loader.height)
    reconst_data_x_by_mean_print = tf.math.sigmoid(reconst_data_x_by_mean_print)
    MNISTData.print_56_pair_images(test_data_x_print, reconst_data_x_by_mean_print, test_label)

    print("Bye~")
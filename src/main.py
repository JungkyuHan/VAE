import tensorflow as tf
from mnist_data import MNISTData
from VAE import VAE


if __name__ == "__main__":
    print("Hi. I am a VAE Tester.")
    latent_vec_dim = 40
    num_epochs = 10
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
    x_train = data_loader.x_train_dataset
    x_tune = data_loader.x_tune_dataset
    opt_adam = tf.keras.optimizers.Adam()
    auto_encoder.train(train_data=x_train, optimizer=opt_adam, num_epochs=num_epochs, tuning_data=x_tune)

    # Calculate mean and sd by using train set
    num_target_items = 10000
    train_data = data_loader.x_train[0:num_target_items]
    train_labels = data_loader.y_train[0:num_target_items]
    train_data = train_data.reshape(num_target_items, data_loader.in_out_dim)
    # train_data_x_print = train_data.reshape(num_target_items, data_loader.width, data_loader.height)

    latent_means, latent_sds, sampled_latent_vec = auto_encoder.encoder.predict(train_data)

    latent_mean_4_label = []
    latent_sd_4_label = []
    num_elements_4_label = []

    for i in range(0, 10):
        latent_mean_4_label.append(tf.zeros(latent_means.shape[1]))
        latent_sd_4_label.append(tf.zeros(latent_sds.shape[1]))
        num_elements_4_label.append(0)

    for (mean, sd, label) in zip(latent_means, latent_sds, train_labels):
        latent_mean_4_label[label] += mean
        latent_sd_4_label[label] += sd
        num_elements_4_label[label] += 1

    for i in range(0, 10):
        latent_mean_4_label[i] /= float(num_elements_4_label[i])
        latent_sd_4_label[i] /= float(num_elements_4_label[i])

    f = open("../model/latent_vec_info.txt", "w")
    for i in range(0, 10):
        f.write("%d, %d\n" % (i, latent_vec_dim))
        mean = latent_mean_4_label[i]
        sd = latent_sd_4_label[i]
        for j in range(0, latent_vec_dim):
            val = mean[j]
            if j < (latent_vec_dim - 1):
                f.write("%.5f, " % val)
            else:
                f.write("%.5f" % val)
        f.write("\n")
        for j in range(0, latent_vec_dim):
            val = sd[j]
            if j < (latent_vec_dim - 1):
                f.write("%.5f, " % val)
            else:
                f.write("%.5f" % val)
        f.write("\n")
    f.close()


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

    save_path = "../model/vae_model.ckpt"
    # save_dir = os.path.dirname(save_path)
    auto_encoder.vae.save_weights(save_path)
    print("load model weights from %s" % save_path)

    print("Bye~")
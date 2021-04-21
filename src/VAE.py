import tensorflow as tf


class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        latent_mean, latent_sd = inputs
        batch_size = tf.shape(latent_mean)[0]
        dim = tf.shape(latent_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch_size, dim))
        sample = tf.math.multiply(latent_sd, epsilon)
        sample = tf.math.add(latent_mean, sample)
        return sample


class VAE:
    def __init__(self, input_output_dim, encoder_common_hidden_layers, encoder_separate_hidden_layers, latent_vec_dim,
                 decoder_hidden_layers, complexity_loss_weight, reconstrcution_loss_weight, batch_size):
        self.encoder = None
        self.decoder = None
        self.vae = None
        self.input_output_dim = input_output_dim
        self.latent_vec_dim = latent_vec_dim
        self.batch_size = batch_size
        self.complexity_loss_weight = complexity_loss_weight
        self.reconstruction_loss_weight = reconstrcution_loss_weight
        af_relu = tf.keras.activations.relu
        af_tanh = tf.keras.activations.tanh
        af_sigmoid = tf.keras.activations.sigmoid

        # Build Encoder
        encoder_input = tf.keras.layers.Input(shape=(self.input_output_dim, ), dtype=tf.float32)
        encoder_inter_layer = encoder_input
        for dim in encoder_common_hidden_layers:
            encoder_inter_layer = tf.keras.layers.Dense(
                units=dim, activation=af_relu, use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros', kernel_regularizer=None,
                bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                bias_constraint=None
            )(encoder_inter_layer)

        latent_mean = encoder_inter_layer
        latent_sd = encoder_inter_layer
        for dim in encoder_separate_hidden_layers:
            latent_mean = tf.keras.layers.Dense(
                units=dim, activation=af_relu, use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros', kernel_regularizer=None,
                bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                bias_constraint=None
            )(latent_mean)

            latent_sd = tf.keras.layers.Dense(
                units=dim, activation=af_relu, use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros', kernel_regularizer=None,
                bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                bias_constraint=None
            )(latent_sd)

        # Encoder output
        latent_mean = tf.keras.layers.Dense(
            units=latent_vec_dim, activation=af_tanh, use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros', kernel_regularizer=None,
            bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
            bias_constraint=None
        )(latent_mean)

        latent_sd = tf.keras.layers.Dense(
            units=latent_vec_dim, activation=af_tanh, use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros', kernel_regularizer=None,
            bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
            bias_constraint=None
        )(latent_sd)

        sampled_latent_vec = Sampling()((latent_mean, latent_sd))
        self.encoder = tf.keras.models.Model(inputs=encoder_input, outputs=[latent_mean, latent_sd, sampled_latent_vec])
        # self.encoder = tf.keras.models.Model(inputs=encoder_input, outputs=[latent_mean, latent_sd])

        # Build Decoder
        decoder_input = tf.keras.layers.Input(shape=(self.latent_vec_dim, ), dtype=tf.float32)
        decoder_hidden_layer = decoder_input
        for dim in decoder_hidden_layers:
            decoder_hidden_layer = tf.keras.layers.Dense(
                units=dim, activation=af_relu, use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros', kernel_regularizer=None,
                bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                bias_constraint=None
            )(decoder_hidden_layer)

        # decoder output
        decoder_output = tf.keras.layers.Dense(
            units=self.input_output_dim, activation=None, use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros', kernel_regularizer=None,
            bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
            bias_constraint=None
        )(decoder_hidden_layer)

        self.decoder = tf.keras.models.Model(inputs=decoder_input, outputs=decoder_output)

        vae_output = self.decoder(sampled_latent_vec)
        self.vae = tf.keras.models.Model(inputs=encoder_input, outputs=vae_output)

    @staticmethod
    def sample_by_reparameterization(latent_mean, latent_sd):
        # re-parametrization trick
        adjust_weight = tf.random.normal(shape=latent_sd.shape, mean=0, stddev=1,
                                         dtype=tf.float32)
        sampled_sd = tf.math.multiply(latent_sd, adjust_weight)
        sampled_latent_vec = tf.math.add(latent_mean, sampled_sd)
        return sampled_latent_vec

    def loss_function(self, true_data, predicted_data, latent_vec_mean, latent_vec_sd):
        # reconstruction
        reconstruction_loss = self.reconstruction_loss_function(true_data, predicted_data)
        # model complexity loss
        complexity_loss = self.complexity_loss_function(latent_vec_mean, latent_vec_sd)

        loss = self.complexity_loss_weight * complexity_loss + self.reconstrcution_loss_weight * reconstruction_loss
        return loss

    def reconstruction_loss_function(self, true_data, predicted_data):
        # reconstruction loss
        reconstruction_loss = tf.nn.sigmoid_cross_entropy_with_logits(true_data, predicted_data)
        reconstruction_loss = tf.math.reduce_sum(reconstruction_loss, axis=[1])
        reconstruction_loss = tf.math.reduce_mean(reconstruction_loss)
        return reconstruction_loss

    def complexity_loss_function(self, latent_vec_mean, latent_vec_sd):
        # model complexity loss
        latent_vec_square_mean = tf.math.square(latent_vec_mean) # square mean
        latent_vec_square_sd = tf.math.square(latent_vec_sd)  # square mean
        latent_vec_log_square_sd = tf.math.log(latent_vec_square_sd)
        ones = tf.ones(shape=latent_vec_mean.shape, dtype=tf.dtypes.float32, name=None)
        complexity_loss = tf.math.add(latent_vec_log_square_sd, ones)
        complexity_loss = tf.math.subtract(complexity_loss, latent_vec_square_sd)
        complexity_loss = tf.math.subtract(complexity_loss, latent_vec_square_mean)
        complexity_loss = -0.5 * tf.math.reduce_sum(complexity_loss, axis=[1])
        complexity_loss = tf.math.reduce_mean(complexity_loss)
        return complexity_loss

    def train(self, train_data, optimizer, num_epochs=1, tuning_data=None):
        train_dataset = train_data.shuffle(buffer_size=1024).batch(self.batch_size)

        for epoch in range(num_epochs):
            loss_sum_in_this_epoch = 0.0
            reconstruction_loss_in_this_epoch = 0.0
            complexity_loss_in_this_epoch = 0.0
            print("\nStart of epoch %d" % (epoch + 1,))
            for step, batch_data in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    latent_vec_means, latent_vec_sds, latent_vec_samples = self.encoder(batch_data, training=True)
                    reconstructed_data = self.decoder(latent_vec_samples, training=True)
                    # loss = self.loss_function(batch_data, reconstructed_data, latent_vec_means, latent_vec_sds)
                    reconstruction_loss = self.reconstruction_loss_function(batch_data, reconstructed_data)
                    complexity_loss = self.complexity_loss_function(latent_vec_means, latent_vec_sds)
                    loss = self.complexity_loss_weight * complexity_loss \
                        + self.reconstruction_loss_weight * reconstruction_loss

                gradients = tape.gradient(loss, self.vae.trainable_weights)
                optimizer.apply_gradients(zip(gradients, self.vae.trainable_weights))

                reconstruction_loss_in_this_epoch += reconstruction_loss
                complexity_loss_in_this_epoch += complexity_loss
                loss_sum_in_this_epoch = self.complexity_loss_weight * complexity_loss_in_this_epoch \
                    + self.reconstruction_loss_weight * reconstruction_loss_in_this_epoch
            print("[Train] loss_sum:%.3f, reconstruction_loss: %.3f, complexity_loss: %.3f" %
                  (loss_sum_in_this_epoch, reconstruction_loss_in_this_epoch, complexity_loss_in_this_epoch))

            if tuning_data is not None:
                tuning_dataset = tuning_data.batch(self.batch_size)
                tuning_loss_in_this_epoch = 0.0
                tuning_reconstruction_loss_in_this_epoch = 0.0
                tuning_complexity_loss_in_this_epoch = 0.0

                for step, batch_tuning_data in enumerate(tuning_dataset):
                    latent_vec_means, latent_vec_sds, latent_vec_samples = self.encoder(batch_tuning_data)
                    reconstructed_data = self.decoder(latent_vec_samples)
                    reconstruction_loss = self.reconstruction_loss_function(batch_tuning_data, reconstructed_data)
                    complexity_loss = self.complexity_loss_function(latent_vec_means, latent_vec_sds)
                    tuning_reconstruction_loss_in_this_epoch += reconstruction_loss
                    tuning_complexity_loss_in_this_epoch += complexity_loss
                    tuning_loss_in_this_epoch = self.complexity_loss_weight * tuning_complexity_loss_in_this_epoch \
                        + self.reconstruction_loss_weight * tuning_reconstruction_loss_in_this_epoch

                print("[Tune] loss_sum:%.3f, reconstruction_loss: %.3f, complexity_loss: %.3f" %
                      (tuning_loss_in_this_epoch, tuning_reconstruction_loss_in_this_epoch,
                       tuning_complexity_loss_in_this_epoch))


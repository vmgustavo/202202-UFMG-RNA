rm(list=ls())

#library(tensorflow)
library(keras)
install_keras()

mnist <- dataset_mnist()



x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y


#plot imagem 2
index_image = 2 ## change this index to see different image.
input_matrix <- x_train[index_image,1:28,1:28]
output_matrix <- apply(input_matrix, 2, rev)
output_matrix <- t(output_matrix)
image(1:28, 1:28, output_matrix, col=gray.colors(256), xlab=paste('Image for digit of: ', y_train[index_image]), ylab="")



# Define a few parameters to be used in the CNN model
batch_size <- 64
num_classes <-10 
epochs <- 5

# Input image dimensions
img_rows <- 28
img_cols <- 28

x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)

#scaling for numerical stability
x_train <- x_train / 255
x_test <- x_test / 255

# Convert class vectors to binary class matrices
#testar com e sem isso
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)


# define model structure 
cnn_model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 8, kernel_size = c(3,3), activation = 'relu', input_shape = input_shape) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 16, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 4, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = num_classes, activation = 'softmax')


summary(cnn_model)


# Compile model
cnn_model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)


# Train model
cnn_history <- cnn_model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  validation_split = 0.2
)

plot(cnn_history)


cnn_model %>% evaluate(x_test, y_test)

# model prediction
cnn_pred <- cnn_model %>% 
  predict(x_test)
head(cnn_pred, n=50)



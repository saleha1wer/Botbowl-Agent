# Neural network - input -> features of state (spacial and non-spacial), output -> win rate
from keras.models import Model
from keras.losses import mean_squared_error, binary_crossentropy
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Input, BatchNormalization, Dropout,Conv3D,Concatenate
import tensorflow as tf
from sklearn.model_selection import train_test_split

def my_custom_loss(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    crossentropy = binary_crossentropy(y_true, y_pred)
    return mse + crossentropy

class Network:
    def __init__(self, input_shape_one, input_shape_two,action_distribution_len):
        self.shape_one = input_shape_one
        self.shape_two = input_shape_two
        self.out_shape = action_distribution_len
        self.network = None

    def initialize_network(self):

        input_one = Input(shape=self.shape_one) # (num_feature_layers, height, width) 
        input_two = Input(shape=self.shape_two) # (113,) 

        conv1 = Conv2D(64,(3,3), padding='same',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_one)
        conv2 = Conv2D(128,(3,3), padding='same',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01))(conv1)
        flat = Flatten()(conv2)
        concatenate = Concatenate()([flat,input_two])
        full = Dense(256, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01))(concatenate)
        batch = BatchNormalization()(full)
        full_1 = Dense(128, activation='relu')(batch)
        full_2 = Dense(64)(full_1)
        full_3 = Dense(32, activation='relu')(full_2)
        full_4 = Dense(16)(full_3)
        out1 = Dense(1,name='win_rate')(full_4)
        out2 = Dense(self.out_shape, activation='softmax',name='action_dis')(batch)
        network = Model(inputs=[input_one, input_two], outputs =[out1,out2], name='Features2WinRate')

        network.compile(
            loss = ['mse','categorical_crossentropy'],
            optimizer = tf.keras.optimizers.Adam(),
            )

        self.network = network
        print(network.summary())

    def _read_data(self, X_spacial, X_non_spacial, win_rates,action_ds):
        # X_spacial = ... # Array of size (N, num_feature_layers, height, width) 
        # X_non_spacial = ... # Array of size (N, 113)
        # win_rates = ...  # Array of size (N,1)

        temp1 = (X_spacial.shape[1], X_spacial.shape[2], X_spacial.shape[3])
        temp2 = (X_non_spacial.shape[1])

        print('shape of data 1: ',temp1)
        print('shape of data 2: ',temp2)

        X_train_non_spacial,X_test_non_spacial,X_train_spacial, X_test_spacial, y_win_train, y_win_test, y_action_train,y_action_test = train_test_split(X_non_spacial,X_spacial, win_rates,action_ds, test_size=0.05) 


        # X_valid_spacial, X_train_spacial, X_valid_non_spacial, X_train_non_spacial = X_train_spacial[:1], X_train_spacial[:], X_train_non_spacial[:2], X_train_non_spacial[2:]         
        # y_win_valid, y_win_train = y_win_train[:1], y_win_train[1:]
        # y_action_valid, y_action_train = y_action_train[:1], y_action_train[1:]

        print('Train: Shape of X1 : {},Shape of X2 : {}, Shape of y1: {}, Shape of y2: {}'.format(X_train_spacial.shape,X_train_non_spacial.shape,y_win_train.shape,y_action_train.shape))
        print('Valid: Shape of X1 : {},Shape of X2 : {},  Shape of y1: {}, Shape of y2: {}'.format(X_test_spacial.shape, X_test_non_spacial.shape,y_win_test.shape,y_action_test.shape))
        # print('Valid: Shape of X1 : {},Shape of X2 : {},  Shape of y1: {}, Shape of y2: {}'.format(X_valid_spacial.shape, X_valid_non_spacial.shape, y_win_valid.shape,y_action_valid.shape))

        # return X_train_spacial, X_train_non_spacial,y_win_train,y_action_train, X_test_spacial, X_test_non_spacial, y_win_test,y_action_test, X_valid_spacial, X_valid_non_spacial, y_win_valid,y_action_valid
        return X_train_spacial, X_train_non_spacial,y_win_train,y_action_train, X_test_spacial, X_test_non_spacial, y_win_test,y_action_test
        
    def train_network(self, X_sp, X_no_sp,y_win,y_action,epochs):
        X_train_sp, X_train_no_sp,y_win_train,y_action_train, X_valid_sp, X_valid_no_sp, y_win_valid,y_action_valid = self._read_data(X_sp, X_no_sp, y_win,y_action)
        self.network.fit(
            x = [X_train_sp, X_train_no_sp],
            y = [y_win_train,y_action_train],
            validation_data = ([X_valid_sp,X_valid_no_sp], [y_win_valid,y_action_valid]),
            epochs = epochs,
            batch_size=4
        )

    def save_network(self,name):
        self.network.save('saved_models/{}.tf'.format(name))




# def read_data(X_spacial, X_non_spacial, win_rates):
#     X_spacial = ... # Array of size (N, num_feature_layers, height, width) 
#     X_non_spacial = ... # Array of size (N, 113)
#     win_rates = ...  # Array of size (N,1)

#     input_shape_one = (X_spacial.shape[1], X_spacial.shape[2], X_spacial.shape[3])
#     input_shape_two = (X_non_spacial.shape[1])

#     print('inp shape 1: ',input_shape_one)
#     print('inp shape 2: ',input_shape_two)

#     X_train_non_spacial,X_test_non_spacial,X_train_spacial, X_test_spacial, y_train, y_test = train_test_split(X_non_spacial,X_spacial, win_rates, test_size=0.2) 

#     X_valid_spacial, X_train_spacial, X_valid_non_spacial, X_train_non_spacial = X_train_spacial[:2400], X_train_spacial[2400:], X_train_non_spacial[:2400], X_train_non_spacial[2400:] 
#     y_valid, y_train = y_train[:2400], y_train[2400:]

#     print('Train: Shape of X1 : {},Shape of X2 : {}, Shape of y : {}'.format(X_train_spacial.shape,X_train_non_spacial.shape,y_train.shape))
#     print('Test: Shape of X1 : {},Shape of X2 : {}, Shape of y : {}'.format(X_test_spacial.shape, X_test_non_spacial.shape,y_test.shape))
#     print('Valid: Shape of X1 : {},Shape of X2 : {}, Shape of y : {}'.format(X_valid_spacial.shape, X_valid_non_spacial.shape, y_valid.shape))



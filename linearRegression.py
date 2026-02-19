import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import time 
np.set_printoptions(precision=2 , suppress=False)

class Linear_regression :
    
    def __init__(self , learning_rate , epochs , x_train , y_train):
        self.learning_rate = learning_rate 
        self.epochs = epochs 
        self.x_train = x_train 
        self.y_train = y_train 
        self.loss = []
        self.epochs_history = []
        
        try:
            self.x_shape = self.x_train.shape[1]
            
        except IndexError :
            self.x_train = self.x_train.reshape(-1 ,1)
            self.x_shape = self.x_train.shape[1]
            
        self.weights = np.zeros((self.x_shape , 1))
        self.intercept = 0
        self.num_row = self.x_train.shape[0]
        
        self.fig = 0 
        self.axes = 0
        
            
    def show(self):
        print(f"Number of epochs : {self.epochs}")
        print(f"the value of learning rate is {self.learning_rate}")
        print(f"the shape of our x train data is {self.x_shape}")
        print(f"Our initialised weights :{self.weights}")
        
    def predict(self , x_data):
        prediction = (x_data @ self.weights) + self.intercept 
        return prediction
    
    
    def mean_squared_error(self , actual_y , prediction ):
        mse = ((actual_y  -  prediction ) **2 ).sum() / self.num_row
        return mse 
    
    
    
    def fit(self):
        
        for _ in range(self.epochs) :
            derivative = 0
            tolerance = 1e-9
            
            # matrix method (is the best way too because it's faster )
            predict = self.predict(self.x_train)
            #print(predict.shape , self.y_train.shape)
            mse = self.mean_squared_error(actual_y =self.y_train  , prediction =predict )
            errors = ( self.x_train @ self.weights + self.intercept   ) - (self.y_train)
            gradients = (2 / self.num_row ) * ( self.x_train.T @ errors )
            self.weights -= self.learning_rate * gradients
            
            # loop method 
            """
            for index, value in enumerate(self.weights, start=0):
                gradient = (( -2 ) /self.num_row) * np.sum((self.y_train.flatten() - predict.flatten()) * self.x_train[:,index])
                self.weights[index] -=  (self.learning_rate * gradient )
                if index == 0 :
                    derivative = gradient
                #here we calculate the derivative of the mse respect to each weight in the array 
            """  
            #updating the intercept 
            gradient_b = (-2/self.num_row) * np.sum(self.y_train - predict)
            self.intercept -= self.learning_rate * gradient_b
            """print(f"the slopes (w) at the {_} is {self.weights}")
            print(f"the intercept (b) in {_} is {self.intercept:.2f}")
            print(f" the derivative of loss respect to the intercept :{gradient_b:.2f}")"""
            #print(f"the mean squared error at epoch {_} is : {mse}")
            self.loss.append(mse)
            self.epochs_history.append(_)
            
            
            if _ > 0 :
                loss_change = abs(self.loss[_] - self.loss[_-1] )
                if loss_change < tolerance :
                    print(f"the training broke in the epoch : {_ + 1 }")
                    break
            if _ % 100 == True :
                pass
                self.visualise(_ ,derivative , mse )
                time.sleep(0.0001)
            
    def enable_visualisation(self ):
        plt.ion()
        self.fig , self.axes = plt.subplots(1 , 2 , figsize=(12,5))
        self.axes[0].scatter(self.x_train[: , 0].flatten() , self.y_train.flatten() ,color="blue" , label="Data" )
        
        self.line , = self.axes[0].plot([], [], color="red", linewidth=2, label="Model")
    
        self.axes[0].set_xlabel("Area")
        self.axes[0].set_ylabel("Price")
        self.axes[0].legend()
        
        # line of loss updating during the training 
        self.loss_line ,  = self.axes[1].plot([] , [] , 'go' , ms=2)
        self.axes[1].set_title("Loss History (Mse)/ gradient")
        self.axes[1].set_label("Epochs")
        
        
        # Prevents labels from overlapping (maytsat7och hhh)
        self.fig.tight_layout()
        
        
            
    def visualise(self, index ,derivative , mse ):
        
        #self.axes[0].cla()
        
        
        x_min, x_max = self.x_train.min(), self.x_train.max()
        x_endpoints = np.array([x_min, x_max])
        y_endpoints = x_endpoints * self.weights.flatten()[0] + self.intercept
        
        #real time  visualisation of fitting the line and loss
        #self.axes[0].scatter(self.x_train[: , 0].flatten() , self.y_train.flatten() ,color="blue" , label="Data" )
        self.line.set_data(x_endpoints, y_endpoints)
        self.axes[0].set_title(f"epoch number {index + 1 } , w={self.weights[0][0]:.3f} , b = {self.intercept:.3f}")
        
        
        epochs_range = range(len(self.loss))
        self.loss_line.set_data(epochs_range , self.loss)
        self.axes[1].set_title(f"Loss History (Mse) = {mse:.4f} / derivative= {derivative:.2f} ")
        self.axes[1].relim()        
        self.axes[1].autoscale_view()
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    
    def denormalize(self , data , mean , std):
        denormalized = (data * std) + mean
        return denormalized 
    
    

data_frame = pd.read_csv("backend/Housing.csv")
data_frame = data_frame.sample(frac=1, random_state=42).reset_index(drop=True)
# we shuffle data because when I trained the model , the mean of the y data is 4000 while the testing mean is 2000 so the r squared is worst than mean guesseer 


x = data_frame[["area" , "bedrooms" , "bathrooms" ]]
y = data_frame[["price"]]

x = np.array(x)
y=np.array(y) / 1000
split_index = int(x.shape[0] *0.8)

x_train , y_train , x_test , y_test = x[:split_index , :] , y[ : split_index] , x[ split_index : , :] , y[split_index :]

x_mean = np.mean(x_train , axis=0)
x_std = np.std(x_train , axis=0)
x_normalized = (x_train - x_mean) / x_std 
x_train = x_normalized




y_mean = np.mean(y_train)
y_std = np.std(y_train)
y_normalized = (y_train - y_mean) / y_std
y_train = y_normalized


lr = Linear_regression(0.0002 , 15000 , x_train ,y_train)




lr.enable_visualisation()
lr.fit()



denomarlized_x = lr.denormalize(data=x_normalized , mean=x_mean , std=x_std)
denomarlized_y = lr.denormalize(data=y_normalized , mean=y_mean , std=y_std)

plt.ioff() # deavtivating the animation mode it won't close our visualisation automatically
"""
columns = ["area" , "bedrooms" , "bathrooms"]
for index in range(3) :
    fig , axes = plt.subplots(1 ,2 , figsize=(12,5))
    predictions = x_normalized[:,index] * lr.weights[index] + lr.intercept
    predictions_denormalized= lr.denormalize(predictions , y_mean , y_std)
    x_endpoints = np.array([denomarlized_x[: ,index].min() , denomarlized_x[: ,index].max()])
    y_endpoints = np.array([predictions_denormalized.min() , predictions_denormalized.max()])
    
    axes[0].cla()
    axes[1].cla()
    axes[0].plot(lr.epochs_history , lr.loss)
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss in each epoch")
    axes[0].legend(["loss"])

    axes[1].scatter(denomarlized_x[:, index] , denomarlized_y , color="blue")
    axes[1].plot(x_endpoints , y_endpoints , linewidth=2 , color='red' , label="Model lr")
    axes[1].set_xlabel(columns[index])
    axes[1].set_ylabel("Price")
    axes[1].set_title("Data")
    plt.show()
"""





#comparing my model from scratch to sklearn 

from sklearn.linear_model import LinearRegression 

sklearn_model = LinearRegression(fit_intercept=True)
sklearn_model.fit(x_train, y_train)

weights = sklearn_model.coef_[0]
bias = sklearn_model.intercept_ 
predictions_sklearn = x_normalized[: ,0] * weights[0] + bias 
denomarlized_x = denomarlized_x[: , 0]
denormalized_prediction_sk = lr.denormalize(data=predictions_sklearn , mean=y_mean , std=y_std)
x_endpoints = np.array([denomarlized_x.min() , denomarlized_x.max()])
y_endpoints = np.array([denormalized_prediction_sk.min() , denormalized_prediction_sk.max()])

#my model calculation 

my_weights = lr.weights
my_bias = lr.intercept 
my_prediction = x_normalized[: , 0] * my_weights[0] + my_bias 
my_prediction = lr.denormalize(my_prediction , y_mean , y_std)
print("skelarn weights and bias are  : " , weights ,bias)
print("my model weights and bias are " , my_weights , my_bias)

fig , axes  = plt.subplots(1 , 2 , figsize = (12,5))

for index in range(2):
     x_endpoints = np.array([denomarlized_x.min() , denomarlized_x.max()])
     axes[index].scatter(x=denomarlized_x , y=denomarlized_y , color="blue")
     axes[index].set_xlabel("Area")
     axes[index].set_ylabel("Price k dollar")
     if index ==0 :
         y_endpoints = np.array([denormalized_prediction_sk.min() , denormalized_prediction_sk.max()])
         axes[index].plot(x_endpoints , y_endpoints , linewidth=2 , color='red' , label="Sklearn Model lr")
         axes[index].set_title("Skelarn model")
     else :
         y_endpoints =  np.array([my_prediction.min() , my_prediction.max()])
         axes[index].plot(x_endpoints , y_endpoints , linewidth=2 , color='red' , label="My model")
         axes[index].set_title("My model")

plt.tight_layout()
plt.show()    



my_mse = lr.mean_squared_error(actual_y=denomarlized_y , prediction=denormalized_prediction_sk)
sklearn_mse = lr.mean_squared_error(actual_y=denomarlized_y , prediction=my_prediction)
absolute_diff = abs(my_mse - sklearn_mse)
percentage_diff = (absolute_diff / sklearn_mse) * 100


print("the mse of sklearn model is :" , sklearn_mse)
print("the mse of my model is : " ,my_mse) 
print(f"The difference is {percentage_diff:.2f}%")

# single prediction 

simple_lr = Linear_regression(0.001, 1500 , x_train[:,0:1] , y_train)
simple_lr.fit()
simple_lr_sk = LinearRegression(fit_intercept=True)
simple_lr_sk.fit(x_train[:,0:] , y_train )
np.savez("backend/House_train_para" , 
         weights = my_weights,
         my_bias = my_bias, 
         sk_weights =weights , 
         sk_bias = bias,
         x_mean = x_mean , 
         x_std = x_std ,
         y_mean = y_mean , 
         y_std = y_std , 
         split_index = split_index,
         simple_weights = simple_lr.weights ,
         simple_intercept = simple_lr.intercept , 
         simple_weight_sk_ = simple_lr_sk.coef_ , 
         simple_inercept_sk_ = simple_lr_sk.intercept_, 
         x_test = x_test  , 
         y_test = y_test)

print(simple_lr.weights)
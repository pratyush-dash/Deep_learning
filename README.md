# CS6910 Assignment 1
 CS6910 Fundamentals of Deep Learning.

Team members: Dip Narayan Gupta(CS21Z025),Monica (CS21Z023)

---

### Instructions to train and evaluate the neural network models:

1. Install the required libraries in your environment using this command:

`
pip install wandb
`

## Question 1

The program, reads the data from `keras.datasets`, picks one example from each class and logs the same to `wandb`.

## Questions 2-4
The neural network is implemented by the class `ANN` 
### Building a `NeuralNetwork`
```Python
    def forward_propagation(self,input)
    def back_propagation(self,X_train,y_train):


```

It can be implemented by passing the following values:

- **layers**  
    An example of layer is as follows:
        def __init__(self,input_layer,hidden_layer,output_layer ,initialisation, activation,loss_function):


- **batch_size**  
    The Batch Size is passed as an integer that determines the size of the mini batch to be taken into consideration.

- **optimizer**  
    The optimizer value is passed as a string, that is internally converted into an instance of the specified optimizer class. 
     steps = 0
      pre_update_w = np.multiply(self.weights,0)
      pre_update_b = np.multiply(self.biases,0)
      update_w = np.multiply(self.weights,0)
      update_b = np.multiply(self.biases,0)
      vw = 0.0
      vb = 0.0
      eps = 1e-8
      a1 =0.0
      gamma = 0.9
      beta = 0.999
      beta1 = 0.9
      beta2 = 0.999
      m_t, v_t, m_hat_w, v_hat_w, m_b,v_b,m_hat_b,v_hat_b = 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 
      
- **intialization**: A string `"Xavier"` can be passed to change the initialization of the weights in the model.

- **epochs**: The number of epochs is passed as an integer to the neural network.(10,15,20)


 

- **X_val**: The validation dataset, used to validate the model.

- **y_val_encode**: `t_val` is the `OneHotEncoded` matrix of the vector `y_val`, of size (10,n), where n is the number of sample.

- **use_wandb**: A flag that lets the user choose whether they want to use wandb for the run or not.
 
- **optimisers*: Optimization parameters to be passed to the optimizers.

### Training the `NeuralNetwork`
The model can be trained by calling the member function: `forward_propogation`, followed by `backward_propogation`. It is done as follows:

```python
self.forward_propogation()
self.backward_propogation()
```

### Testing the `NeuralNetwork`
The model can be tested by calling the `predict` member function, with the testing dataset and the expected `y_test`. The `y_test` values are only used for calculating the test accuracy. It is done in the following manner:

```python
accuracy , test_labels = model.predict(X_test, t_test)
```



import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


# Creating a function f(X) with a slope of -5
X = torch.arange(-5, 5, 0.1).view(-1, 1)
func = -5 * X + 42.0
# Adding Gaussian noise to the function f(X) and saving it in Y
Y = func + 2.0 * torch.randn(X.size())

# Plot and visualizing the data points in blue
plt.plot(X.numpy(), Y.numpy(), 'b+', label='Y')
plt.plot(X.numpy(), func.numpy(), 'r', label='func')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid('True', color='y')
plt.show()


class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        predict_y = self.linear(x)
        return predict_y


learning_rate = 0.05
loss_list = []
iteration = 50

linear_model = LinearRegression()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(linear_model.parameters(), lr=learning_rate)

for i in range(iteration):
    # zero the parameter gradients
    optimizer.zero_grad()
    # making predictions with forward pass
    Y_pred = linear_model(X)
    # calculating the loss between original and predicted data points
    loss = criterion(Y_pred, Y)
    # backward + optimize
    loss.backward()
    optimizer.step()
    # printing the values for understanding
    loss_list.append(loss.item())
    print('{},\tloss:{}'.format(i, loss.item()))

# Plotting the loss after each iteration
plt.plot(loss_list, 'r')
plt.tight_layout()
plt.grid('True', color='y')
plt.xlabel("Epochs/Iterations")
plt.ylabel("Loss")
plt.show()

x_test = 4.0
test_variable = Variable(torch.Tensor([[x_test]]))
predict_y = linear_model(test_variable)
print("The result of predictions after training x={}, y_pred={}, y_true={}, as [-5 * x + 42]".format(x_test, linear_model(test_variable).item(), -5 * x_test + 42))

for param in linear_model.parameters():
    print(param, param.size())

model_scripted = torch.jit.script(linear_model) # Export to TorchScript
model_scripted.save('model_scripted.pt') # Save

################ vs sklearn #############################
# from sklearn.linear_model import LinearRegression as LR
# reg = LR().fit(X.numpy(), Y.numpy())
# print(reg.coef_, reg.intercept_)
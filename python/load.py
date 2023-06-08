import torch
from torch.autograd import Variable

model = torch.jit.load('model_scripted.pt')
model.eval()

x_test = 4.0
test_variable = Variable(torch.Tensor([[x_test]]))
predict_y = model(test_variable)
print("The result of predictions after training x={}, y_pred={}, y_true={}, as [-5 * x + 42]"
      .format(x_test, model(test_variable).item(), -5 * x_test + 42)
      )

# External dependencies
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 
from pyDOE import lhs 


# Initialize random number generators
torch.manual_seed(1234)
np.random.seed(1234)

# Set default data types for PyTorch
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=8)
torch.set_float32_matmul_precision('high')

#%% Network

# Set device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Working on {device}")

# Create a custom module for the activation function
class SecondDegreeActivation(nn.Module):
    def __init__(self, init_exp=2.0 ):
        super(SecondDegreeActivation, self).__init__()
        self._exp = nn.Parameter(torch.tensor(init_exp))

    def forward(self, x):
        ex = self._exp
        teps = torch.tensor(0.00001)
        
        return torch.pow(x**2+0.000001,    ( ex )   )

# Neural network class definition
class Net(nn.Module):
    def __init__(self, layers, 
                 xs_bot_edge, ys_bot_edge,
                 xs_top_edge, ys_top_edge,
                 xs_left_edge, ys_left_edge,
                 xs_right_edge, ys_right_edge,
                 x_domain, y_domain):
        
        super(Net, self).__init__() 
              
        self.layers = len(layers)
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        
        self.xs_bot_edge = xs_bot_edge
        self.ys_bot_edge = ys_bot_edge
        self.xs_top_edge = xs_top_edge
        self.ys_top_edge = ys_top_edge
        self.xs_left_edge = xs_left_edge
        self.ys_left_edge = ys_left_edge
        self.xs_right_edge = xs_right_edge
        self.ys_right_edge = ys_right_edge
        
        self.x_domain = x_domain
        self.y_domain = y_domain        
        
        self.optimizer = None
        self.train_loss_history = []

        self.second_degree_activation = SecondDegreeActivation()

    def forward(self, X):
        if torch.is_tensor(X) != True:         
            X = torch.from_numpy(X)                
        
        a0, a1 = torch.chunk(X.float(),2)
        a0 = torch.sin(a0)
        a1 = torch.sin(a1/1)
        a = torch.cat([a0,a1], dim=0)
        prelu_a = torch.tensor(1.0)

        for i in range(self.layers - 2):           
            z = self.linears[i](a)    
            a = self.second_degree_activation(z)
            
            if i == 1:
                x1 = a
            
        a = self.linears[-1](a) 
        return a
    
    def network_prediction(self, x, y):
        a  =  self.forward(torch.cat((x, y), 1))
        return a[:,0], a[:,1], a[:,2], a[:,3], a[:,4]
    
    def PDE_residual(self, x, y):
        mu = 0.2
        lamb = 1.0
        # Compute the differential equation
        u, v, mixSigXX, mixSigYY, mixSigXY = self.network_prediction(x, y)
        u_x = self.get_derivative(u, x, 1)
        u_y = self.get_derivative(u, y, 1)
        v_x = self.get_derivative(v, x, 1)
        v_y = self.get_derivative(v, y, 1)

        strainxx = 1/2*(u_x + u_x)
        strainxy = 1/2*(u_y + v_x)
        strainyy = 1/2*(v_y + v_y)

        sigxx = mu * 2 * u_x + lamb * (u_x + v_y)
        sigyy = mu * 2 * v_y + lamb * (u_x + v_y)
        sigxy = mu * (v_x + u_y)

        mixSigXX = mixSigXX.unsqueeze(1)
        mixSigYY = mixSigYY.unsqueeze(1)
        mixSigXY = mixSigXY.unsqueeze(1)

        mixSigXX_x = self.get_derivative(mixSigXX, x, 1)
        mixSigXY_x = self.get_derivative(mixSigXY, x, 1)
        mixSigXY_y = self.get_derivative(mixSigXY, y, 1)
        mixSigYY_y = self.get_derivative(mixSigYY, y, 1)
        resid = (mixSigXX_x + mixSigXY_y)**2 + (mixSigXY_x + mixSigYY_y)**2 + (mixSigXX-sigxx)**2 + (mixSigXY-sigxy)**2 + (mixSigYY-sigyy)**2

        return resid
        #return strainxx * sigxx + strainyy * sigyy + strainxy * sigxy
    
    def get_derivative(self, y, x, n):
        # General formula to compute the n-th order derivative of y = f(x) with respect to x
        if n == 0:
            return y
        else:
            dy_dx = torch.autograd.grad(y, x, torch.ones_like(y).to(device), create_graph=True, retain_graph=True, allow_unused=True)[0]
        return self.get_derivative(dy_dx, x, n - 1)

    def loss_BC(self):
        mu = 0.2
        lamb = 1.0

        # left edge Neumann BC
        # Compute the differential equation
        ul, vl, *_ = self.network_prediction(self.xs_left_edge, self.ys_left_edge)
        ul_x = self.get_derivative(ul, self.xs_left_edge, 1)
        ul_y = self.get_derivative(ul, self.ys_left_edge, 1)
        vl_x = self.get_derivative(vl, self.xs_left_edge, 1)
        vl_y = self.get_derivative(vl, self.ys_left_edge, 1)

        sigxx_left = mu * 2 * ul_x + lamb * (ul_x + vl_y)
        mse_BC_left = torch.mean(( ul )**2)

        # right edge Neumann BC
        # Compute the differential equation
        u, v, mixSigXX, mixSigYY, mixSigXY = self.network_prediction(self.xs_right_edge, self.ys_right_edge)
        u_x = self.get_derivative(u, self.xs_right_edge, 1)
        u_y = self.get_derivative(u, self.ys_right_edge, 1)
        v_x = self.get_derivative(v, self.xs_right_edge, 1)
        v_y = self.get_derivative(v, self.ys_right_edge, 1)

        sigxx_right = mu * 2 * u_x + lamb * (u_x + v_y)
        mse_BC_right = torch.mean(mixSigXX ** 2) #traction on right face

        #bottom boundary fixed
        u_pred_BC1, v_pred_BC1, *_  = self.network_prediction(self.xs_bot_edge, self.ys_bot_edge)
        mse_BC_bot = torch.mean((v_pred_BC1 )**2) + torch.mean((u_pred_BC1 )**2)
        
        #top boundary moved up 0.2
        u_pred_BC2, v_pred_BC2, *_ = self.network_prediction(self.xs_top_edge, self.ys_top_edge)
        mse_BC_top = torch.mean((v_pred_BC2 - 0.1)**2) + torch.mean((u_pred_BC2 - 0.0)**2)
        
        mse_BC = mse_BC_bot + mse_BC_top + mse_BC_left + mse_BC_right
        return mse_BC

    def loss_interior(self):
        pde_resid = self.PDE_residual(self.x_domain, self.y_domain)
        mse_interior = torch.mean((pde_resid)**1)
        
        return mse_interior

    def loss_func(self):

        mse_BC = self.loss_BC()
        mse_domain = self.loss_interior()

        return mse_BC, mse_domain

    def closure(self):
        self.optimizer.zero_grad()
        mse_b, mse_f = self.loss_func()
        total_loss =  mse_b + mse_f
        total_loss.backward(retain_graph=True)
        return total_loss

    def train(self, epochs, optimizer='Adam', **kwargs):

        if optimizer=='Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), **kwargs)

        elif optimizer=='L-BFGS':
            self.optimizer = torch.optim.LBFGS(self.parameters(), **kwargs)

        # Training loop
        for epoch in range(epochs):
            mse_BC, mse_domain = self.loss_func()
            total_loss =  mse_BC + mse_domain
            
            self.train_loss_history.append([total_loss.cpu().detach(), mse_BC.cpu().detach(), mse_domain.cpu().detach()])

            self.optimizer.step(self.closure)

            if epoch % 100 == 0:
                print(f'Epoch ({optimizer}): {epoch}, Total Loss: {total_loss.detach().cpu().numpy()}')
            
    def get_training_history(self):
        loss_his = np.array(self.train_loss_history)
        total_loss, mse_BC, mse_domain = np.split(loss_his, 4, axis=1)
        return total_loss, mse_BC, mse_domain

#%% Data

# x and y each run from 0 to 1.0
# hold y=0 fixed in x and y
# pull y=1 up by 1 u (vertical), v held 0 (horizontal)
nx = ny = 50
x = np.linspace(0.0, 1.0, nx).reshape(-1,1) # Space x domain
y = np.linspace(0.0, 1.0, ny).reshape(-1,1) # Space y domain

X, Y = np.meshgrid(x[:, 0], y[:, 0]) # space domain, 2, -2

# Boundary min y
bottom_edge_xs = x
bottom_edge_ys = np.zeros(x.shape[0])

# boundary max y
top_edge_xs = x
top_edge_ys = np.ones(x.shape[0])*1.0


#left edge, min x
left_edge_xs = np.zeros(y.shape[0])
left_edge_ys = y

#left edge, min x
right_edge_xs = np.ones(y.shape[0])
right_edge_ys = y

# Create collocation points with latin hypercube sampling
# Lower and upper bound of the space domain
lb = np.zeros((1,2))
ub = np.zeros((1,2))
lb[0, 0] = x[0]; lb[0, 1] = y[0]
ub[0, 0] = x[-1]; ub[0, 1] = y[-1]

# Number of Points to sample for lh
#NPoints_domain = 10000
#X_Y_domain = lb + (ub - lb) * lhs(2, NPoints_domain)

#interior points
nx = ny = 50
x = np.linspace(0.0, 1.0, nx).reshape(-1,1) # Space x domain
y = np.linspace(0.0, 1.0, ny).reshape(-1,1) # Space y domain

X, Y = np.meshgrid(x[1:-1, 0], y[1:-1, 0]) # space domain, 2, -2
X_flat = X.flatten() 
Y_flat = Y.flatten() 
X_Y_domain = np.vstack((X_flat, Y_flat)).T

#%% Network

layers = [2, 16,16, 5]

x_tens_bot_edge = torch.tensor(bottom_edge_xs[:], requires_grad=True).view(-1,1).float().to(device)
y_tens_bot_edge = torch.tensor(bottom_edge_ys[:], requires_grad=True).view(-1,1).float().to(device)

x_tens_top_edge = torch.tensor(top_edge_xs[:], requires_grad=True).view(-1,1).float().to(device)
y_tens_top_edge = torch.tensor(top_edge_ys[:], requires_grad=True).view(-1,1).float().to(device)

x_tens_left_edge = torch.tensor(left_edge_xs[:], requires_grad=True).view(-1,1).float().to(device)
y_tens_left_edge = torch.tensor(left_edge_ys[:], requires_grad=True).view(-1,1).float().to(device)

x_tens_right_edge = torch.tensor(right_edge_xs[:], requires_grad=True).view(-1,1).float().to(device)
y_tens_right_edge = torch.tensor(right_edge_ys[:], requires_grad=True).view(-1,1).float().to(device)

x_domain = torch.tensor(X_Y_domain[:, 0], requires_grad=True).view(-1,1).float().to(device)
y_domain = torch.tensor(X_Y_domain[:, 1], requires_grad=True).view(-1,1).float().to(device)

net = Net(layers,  
                 x_tens_bot_edge, y_tens_bot_edge, 
                 x_tens_top_edge, y_tens_top_edge, 
                 x_tens_left_edge, y_tens_left_edge,
                 x_tens_right_edge, y_tens_right_edge,
                 x_domain, y_domain).to(device)

# Define the Xavier initialization function
def xavier_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

net.apply(xavier_init)

def weights_init_normal(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

#%% Train
net.train(400, optimizer='Adam', lr=5e-2) #2500
net.train(400, optimizer='L-BFGS') #500

#%% plots

x_domain = torch.tensor(X[:], requires_grad=True).view(-1,1).float().to(device)
y_domain = torch.tensor(Y[:], requires_grad=True).view(-1,1).float().to(device)

u, v, mixSigXX, mixSigYY, mixSigXY = net.network_prediction(x_domain, y_domain)
resid = net.PDE_residual(x_domain, y_domain)
plt.figure(figsize=(8, 6))
u_npArr = u.detach().numpy().reshape(X.shape[0],Y.shape[0])
v_npArr = v.detach().numpy().reshape(X.shape[0],Y.shape[0])
resid_npArr = resid.detach().numpy().reshape(X.shape[0],Y.shape[0])
plt.scatter(X+u_npArr, Y+v_npArr,c = resid_npArr, cmap='viridis')
# plt.scatter(X+u_npArr, Y+v_npArr)
plt.colorbar(label='Residual')
plt.axis('equal')
plt.show()


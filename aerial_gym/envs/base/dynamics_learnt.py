import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from aerial_gym.envs.base.dynamics_flightmare import FlightmareDynamics



class LearntDynamics(nn.Module, FlightmareDynamics):

    def __init__(self, device, param_file_path='', initial_params={}):
        FlightmareDynamics.__init__(self, initial_params)
        super(LearntDynamics, self).__init__()
        torch.cuda.set_device(self.device)

        # Action transformation parameters
        
        self.linear_at = nn.Parameter(
            torch.diag(torch.ones(4)), requires_grad=True
        ).to(device)
        self.linear_st = nn.Parameter(
            torch.diag(torch.ones(12)), requires_grad=True
        ).to(device)
        self.linear_state_1 = nn.Linear(16, 64).to(device)
        torch.nn.init.normal_(self.linear_state_1.weight, mean=0, std=1)
        torch.nn.init.normal_(self.linear_state_1.bias, mean=0, std=1)

        self.linear_state_2 = nn.Linear(64, 12).to(device)
        torch.nn.init.normal_(self.linear_state_2.weight, mean=0, std=1)
        torch.nn.init.normal_(self.linear_state_2.bias, mean=0, std=1)
        # print("FFFFFFFFFFFFFFFFFFFFFFFUCKKKKKKKKKKKKKKKKKKKKKKK")
        # print(self.linear_state_1.weight.device, self.linear_state_1.bias.device)

        # VARIABLES - dynamics parameters
        # self.torch_translational_drag = torch.Variable(torch.from_numpy(
        #     self.copter_params.translational_drag
        # ))
        # self.torch_rotational_drag = torch.Variable(torch.from_numpy(
        #     self.copter_params.rotational_drag
        # ))
        self.mass = nn.Parameter(
            torch.tensor([self.mass]),
            requires_grad=True  # , name="mass"
        ).to(device)
        self.torch_inertia_vector = nn.Parameter(
            torch.from_numpy(self.inertia_vector).float(),
            requires_grad=True,
            # name="inertia"
        ).to(device)
        self.torch_kinv_vector = nn.Parameter(
            torch.tensor(self.kinv_ang_vel_tau).float(),
            requires_grad=True,
            # name="kinv"
        ).to(device)

        # derivations from params
        self.torch_inertia_J = torch.diag(self.torch_inertia_vector)
        self.torch_kinv_ang_vel_tau = torch.diag(self.torch_kinv_vector)

        self.param_file_path = param_file_path
        self.device = device


    def state_transformer(self, state, action):
        state_action = torch.cat((state, action), dim=1)
        layer_1 = torch.relu(self.linear_state_1(state_action)).to(self.device)
        new_state = self.linear_state_2(layer_1)
        # TODO: activation function?
        return new_state

    def forward(self, state, action, dt):
        # print(self.linear_at.device, action.device)
        # print("In dynamic--------------------------")
        # print("action:", action.device)
        # print("state", state.device)
        # print("linear_at", self.linear_at.device)
        action_transformed = torch.matmul(
            self.linear_at, torch.unsqueeze(action, 2)
        )[:, :, 0]
        # print("action_transformed", action_transformed.device)
        # print("self.linear_st", self.linear_st.device)
        # print(action.shape, self.linear_st.shape)
        state_transformed = torch.matmul(
            self.linear_st, torch.unsqueeze(state, 2)
        )[:, :, 0]
        # state_transformed = torch.matmul(
        #     action, torch.unsqueeze(state, 2)
        # )[:, :, 0]
        # print("state_transformed", state_transformed.device)
        # run through D1
        new_state = self.simulate_quadrotor(action_transformed, state_transformed, dt)
        # print("new_state", new_state.device)
        # run through T
        # print("??????????????????????????????????????????????????????")
        # print(state_transformed.shape, self.linear_st.shape, state_transformed.shape)
        added_new_state = self.state_transformer(state, action_transformed)
        # print(added_new_state.shape, state_transformed.shape)
        # print("added_new_state", added_new_state.device)
        # print(new_state.device, added_new_state.device)
        return new_state + added_new_state
        # return state_transformed

    def save_parameters(self):
        """
        Save the parameter of Model to file
        :param file_path: path of parameter saving file
        """
        state_dict = {
            'linear_at': self.linear_at.data,
            'linear_st': self.linear_st.data,
            'linear_state_1': self.linear_state_1.state_dict(),
            'linear_state_2': self.linear_state_2.state_dict(),
            'mass': self.mass.data,
            'torch_inertia_vector': self.torch_inertia_vector.data,
            'torch_kinv_vector': self.torch_kinv_vector.data,
        }
        torch.save(state_dict, self.param_file_path)
        print(f"Learnable dynamics parameters saved to {self.param_file_path}")

    def load_parameters(self):
        """
        Load the parameter of model from file
        :param file_path: path to load the file
        """
        state_dict = torch.load(self.param_file_path, map_location=self.device)
        self.linear_at.data = state_dict['linear_at']
        self.linear_st.data = state_dict['linear_st']
        self.linear_state_1.load_state_dict(state_dict['linear_state_1'])
        self.linear_state_2.load_state_dict(state_dict['linear_state_2'])
        self.mass.data = state_dict['mass']
        self.torch_inertia_vector.data = state_dict['torch_inertia_vector']
        self.torch_kinv_vector.data = state_dict['torch_kinv_vector']

        self.linear_at = self.linear_at.to(self.device)
        self.linear_st = self.linear_st.to(self.device)
        self.mass = self.mass.to(self.device)
        self.torch_inertia_vector = self.torch_inertia_vector.to(self.device)
        self.torch_kinv_vector = self.torch_kinv_vector.to(self.device)

        print(f"Learnable dynamics parameters loaded from {self.param_file_path}")

        
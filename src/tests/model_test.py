from copy import deepcopy
from models import actor, critic, model
import torch
import torch.nn as nn
import numpy as np
from utilities.utils import t

hyper_ps = {
    'c_hidden_size': 5,
    'c_hidden_layers': 2,
    'a_hidden_size': 4,
    'a_hidden_layers': 1,
    'c_learning_rate': 0.01,
    'c_momentum': .9,
    'a_learning_rate': 0.01,
    'a_momentum': .9,
    'action_dim': 3,
    'state_dim' : 2,
    'batch_size' : 25,
}

hidden_size = hyper_ps['a_hidden_size']
hidden_layers = hyper_ps['a_hidden_layers']
action_dim = hyper_ps['action_dim']

mse_loss = torch.nn.MSELoss()

model_object = model.Model()
actor_net = actor.Actor()
critic_net = critic.Critic()

x_critic = None

x_actor = torch.rand((hyper_ps['batch_size'], hyper_ps['state_dim']))
x_actor2 = torch.rand((hyper_ps['batch_size'], hyper_ps['state_dim']))

example_actions = torch.rand((hyper_ps['batch_size'], hyper_ps['action_dim']))

target_q = torch.rand((hyper_ps['batch_size'], 1))

y_actor = None
y_critic = None


def test_model():

    test_init()
    test_actor_set_params()
    test_critic_set_params()
    test_forward_actor()
    test_forward_critic()
    test_critic_loss()
    test_actor_backward_pass()


def test_init():

    assert isinstance(model_object, nn.Module)
    assert isinstance(actor_net, nn.Module)
    assert isinstance(critic_net, nn.Module)

    assert hasattr(actor_net, 'fc_base')
    assert hasattr(actor_net, 'fc_logsd')
    assert hasattr(actor_net, 'fc_mean')
    assert hasattr(actor_net, 'optimiser')
    assert hasattr(actor_net, 'evaluate')


    assert hasattr(critic_net, 'fc')
    assert hasattr(critic_net, 'criterion')
    assert hasattr(critic_net, 'optimiser')
    assert hasattr(critic_net, 'evaluate')
    

    assert isinstance(critic_net.criterion, torch.nn.MSELoss)

    print("Actor and critic networks initialized correctly")

def test_actor_set_params():

    actor_net.set_params(hyper_ps)

    assert isinstance(actor_net.fc_base, nn.Sequential)
    assert isinstance(actor_net.fc_mean, nn.Linear)
    assert isinstance(actor_net.fc_logsd, nn.Linear)
    assert isinstance(actor_net.optimiser, torch.optim.SGD)

    print("Actor network's set_params has run successfully")

def test_critic_set_params():

    critic_net.set_params(hyper_ps)

    assert isinstance(critic_net.fc, nn.Sequential)
    assert isinstance(critic_net.optimiser, torch.optim.SGD)

    print("Critic network's set_params has run successfully")
  
def test_forward_actor():

    normal, y_actor = actor_net.forward(x_actor)
    assert y_actor.shape == (hyper_ps['batch_size'], hyper_ps['action_dim'])

    assert isinstance(normal, torch.distributions.Normal)
        
    x = actor_net.fc_base(x_actor2)
    assert (normal.mean.shape == actor_net.fc_mean(x).shape)

    sd = actor_net.fc_logsd(x).exp()
    assert (normal.stddev.shape == sd.shape)

    print("Actor forward returns action of correct dimensions and normal of correct mean and variance dimensions")

def test_forward_critic():

    y_critic = critic_net.forward(x_actor)
    assert y_critic.shape == (hyper_ps['batch_size'], 1)

    print("Crtic forward returns q-value of correct dimensions")

def test_critic_loss():

    y_critic = critic_net.forward(x_actor)
    loss = critic_net.backward(y_critic, target_q)
    assert loss != 0.0 # unsure about this test
    print("Crtic backprop successful")

def test_actor_backward_pass():

    y_critic = critic_net.forward(x_actor)
    loss = critic_net.backward(y_critic, target_q)

    state_values = np.array(critic_net.evaluate(x_actor).squeeze(1))
    dummy_adv = t(1000 - state_values, device='cpu')

    normal, _ = actor_net.forward(x_actor)
    log_pis = normal.log_prob(example_actions)
    log_pis = torch.sum(log_pis, dim=1)

    losses = -log_pis * dummy_adv
    losses = losses / hyper_ps['batch_size']

    actor_net.backward(losses, device='cpu')

    mean_loss = torch.sum(losses)

    assert mean_loss != 0.0 # unsure about this test

    print("Actor backprop successful")
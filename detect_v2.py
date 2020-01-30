import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from utils_v2 import Noisy,random_label

noisy = Noisy.apply


""" Return the value of l1 norm of [img] with noise radius [n_radius]"""
def l1_detection(model, 
                 img,
                 n_radius):
    return torch.norm(F.softmax(model(img)) - F.softmax(model(noisy(img, n_radius))), 1).item()

""" Return the number of steps to cross boundary using targeted attack on [img]. Iteration stops at [cap] steps """
def targeted_detection(model, 
                       img,
                       lr, 
                       t_radius, 
                       cap=200,
                       margin=20,
                       use_margin=False):
    model.eval()
    x_var = torch.autograd.Variable(img.clone().cuda(), requires_grad=True)
    true_label = model(x_var.clone()).data.max(1, keepdim=True)[1][0].item()
    optimizer_s = optim.SGD([x_var], lr=lr)
    target_l = torch.LongTensor([random_label(true_label, dataset=dataset)]).cuda()
    counter = 0
    while model(x_var.clone()).data.max(1, keepdim=True)[1][0].item() == true_label:
        optimizer_s.zero_grad()
        output = model(x_var)
        if use_margin:
            target_l = target_l[0].item()
            _, top2_1 = output.data.cpu().topk(2)
            argmax11 = top2_1[0][0]
            if argmax11 == target_l:
                argmax11 = top2_1[0][1]
            loss = (output[0][argmax11] - output[0][target_l] + margin).clamp(min=0)
        else:
            loss = F.cross_entropy(output, target_l)
        loss.backward()

        x_var.data = torch.clamp(x_var - lr * x_var.grad.data, min=0, max=1)
        x_var.data = torch.clamp(x_var - img, min=-t_radius, max=t_radius) + img
        counter += 1
        if counter >= cap:
            break
    return counter

""" Return the number of steps to cross boundary using untargeted attack on [img]. Iteration stops at [cap] steps """
def untargeted_detection(model, 
                         img, 
                         lr, 
                         u_radius, 
                         cap=1000,
                         margin=20,
                         use_margin=False):
    model.eval()
    x_var = torch.autograd.Variable(img.clone().cuda(), requires_grad=True)
    true_label = model(x_var.clone()).data.max(1, keepdim=True)[1][0].item()
    optimizer_s = optim.SGD([x_var], lr=lr)
    counter = 0
    while model(x_var.clone()).data.max(1, keepdim=True)[1][0].item() == true_label:
        optimizer_s.zero_grad()
        output = model(x_var)
        if use_margin:
            _, top2_1 = output.data.cpu().topk(2)
            argmax11 = top2_1[0][0]
            if argmax11 == true_label:
                argmax11 = top2_1[0][1]
            loss = (output[0][true_label] - output[0][argmax11] + margin).clamp(min=0)
        else:
            loss = -F.cross_entropy(output, torch.LongTensor([true_label]).cuda())
        loss.backward()

        x_var.data = torch.clamp(x_var - lr * x_var.grad.data, min=0, max=1)
        x_var.data = torch.clamp(x_var - img, min=-u_radius, max=u_radius) + img
        counter += 1
        if counter >= cap:
            break
    return counter

""" Return a set of values of l1 norm. 
    [attack] can be 'real' or any attack name you choose
    [real_dir] specifies the root directory of real images
    [adv_dir] specifies the root directory of adversarial images
    [lowind] specifies the lowest index of the image to be sampled
    [upind] specifies the highest index of the image to be sampled
    [title] specifies the type of attack
    [n_radius] specifies the noise radius
"""
def l1_vals(model, 
            dataloader, 
            attack, 
            n_radius):
    vals = np.zeros(0)
    model.eval()
    if attack == "Clean":
        for img, name in dataloader:
            #cause the batch size is one
            #if you are using own data, uncomment following lines to make sure only detect images which are correctly classified
            view_data_label = int(name[0].split('_')[-1][1])
            predicted_label = model(img.clone()).data.max(1, keepdim=True)[1][0]
            if predicted_label != view_data_label:
                continue  # note that only load images that were classified correctly
            val = l1_detection(model, img, n_radius)
            vals = np.concatenate((vals, [val]))
    else:
        count = len(dataloader.dataset)
        for img, name in dataloader:
            #cause the batch size is one
            real_label = int(name[0].split('_')[-2][1])
            predicted_label = model(img.clone()).data.max(1, keepdim=True)[1][0]
            if real_label == predicted_label:
                count -= 1#number of successful adversary minus 1
                continue#only load successful adversary
            val = l1_detection(model, img, n_radius)
            vals = np.concatenate((vals, [val]))
        print('this is number of success in l1 detection', count)
    return vals

""" Return a set of number of steps using targeted detection.
    [attack] can be 'real' or any attack name you choose.
    [real_dir] specifies the root directory of real images.
    [adv_dir] specifies the root directory of adversarial images.
    [lowind] specifies the lowest index of the image to be sampled.
    [upind] specifies the highest index of the image to be sampled.
    [title] specifies the type of attack.
    [targeted_lr] specifies the learning rate for targeted attack detection criterion.
    [t_radius] specifies the radius of targeted attack detection criterion.
"""
def targeted_vals(model, 
                  dataloader, 
                  attack, 
                  targeted_lr, 
                  t_radius):
    vals = np.zeros(0)
    model.eval()
    if attack == "Clean":
        for img, name in dataloader:
            #if you are using own data, uncomment following lines to make sure only detect images which are correctly classified
            #cause the batch size is one
            view_data_label = int(name[0].split('_')[-1][1])
            predicted_label = model(img.clone()).data.max(1, keepdim=True)[1][0]
            if predicted_label != view_data_label:
                continue  # note that only load images that were classified correctly
            val = targeted_detection(model, img, targeted_lr, t_radius)
            vals = np.concatenate((vals, [val]))
    else:
        count = len(dataloader.dataset)
        for img, name in dataloader:
            #cause the batch size is one
            real_label = int(name[0].split('_')[-2][1])
            predicted_label = model(img.clone()).data.max(1, keepdim=True)[1][0]
            if real_label == predicted_label:
                count -= 1#number of successful adversary minus 1
                continue#only load successful adversary
            val = targeted_detection(model, img, targeted_lr, t_radius)
            vals = np.concatenate((vals, [val]))
        print('this is number of success in targeted detection', count)
    return vals


""" Return a set of number of steps using untargeted detection. 
    [attack] can be 'real' or any attack name you choose.
    [real_dir] specifies the root directory of real images.
    [adv_dir] specifies the root directory of adversarial images.
    [lowind] specifies the lowest index of the image to be sampled.
    [upind] specifies the highest index of the image to be sampled.
    [title] specifies the type of attack.
    [untargeted_lr] specifies the learning rate for untargeted attack detection criterion.
    [u_radius] specifies the radius of untargeted attack detection criterion.
"""
def untargeted_vals(model, 
                    dataloader,
                    attack, 
                    untargeted_lr, 
                    u_radius):
    vals = np.zeros(0)
    model.eval()
    if attack == "Clean":
        for img, name in dataloader:
            #if you are using own data, uncomment following lines to make sure only detect images which are correctly classified
            #cause the batch size is one
            view_data_label = int(name[0].split('_')[-1][1])
            predicted_label = model(img.clone()).data.max(1, keepdim=True)[1][0]
            if predicted_label != view_data_label:
                continue  # note that only load images that were classified correctly
            val = untargeted_detection(model, img, untargeted_lr, u_radius)
            vals = np.concatenate((vals, [val]))
    else:
        count = len(dataloader.dataset)
        for img, name in dataloader:
            #cause the batch size is one
            real_label = int(name[0].split('_')[-2][1])
            predicted_label = model(img.clone()).data.max(1, keepdim=True)[1][0]
            if real_label == predicted_label:
                count -= 1#number of successful adversary minus 1
                continue#only load successful adversary
            val = untargeted_detection(model, img, untargeted_lr, u_radius)
            vals = np.concatenate((vals, [val]))
        print('this is number of success in untargeted detection', cout)
    return vals

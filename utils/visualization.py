import torch
import torch.nn as nn
import numpy as np
import cv2
import os


# visualizes Grad-CAM
# input: SINGLE image (C, H, W)
# shows neuron activations given EACH LABEL
def gradcam(net, out, label):
    net.zero_grad()
    out['logits'][0, label].backward(retain_graph=True)
    grad = net.gradients[0].cpu().data.numpy()
    del net.gradients
    feature = out['gradcam_activation'][0].cpu().data.numpy()
    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(feature.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * feature[i, :, :]
    return (label, cam)


def visualize_gradcam(image, image_key, grad_cams, out_dir, class_names):
    fname = os.path.join(out_dir, '%s.jpg' % image_key)
    # denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).cuda().unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).cuda().unsqueeze(-1).unsqueeze(-1)
    # (X / 255 - mean) / std = Y --> (Y * std + mean) * 255 = X
    image = ((image * std + mean) * 255).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)[:, :, [2, 1, 0]]
    cv2.imwrite(fname, image)
    label, cam = grad_cams
    fname = os.path.join(out_dir, 'gradcam_%s_%s.jpg' % (image_key, class_names[label]))
    feature = np.maximum(cam, 0)
    feature = cv2.resize(feature, image.shape[:2])
    feature = feature / (np.max(feature) + 1e-12)
    heatmap = cv2.applyColorMap((feature * 255.).astype(np.uint8), cv2.COLORMAP_JET)
    merged = (image.astype(np.float) * 0.6 + heatmap.astype(np.float) * 0.4).astype(np.uint8)
    cv2.imwrite(fname, merged)


def visualize_feature(image, image_key, featuredict, out_dir, label, class_names):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(2).unsqueeze(3)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(2).unsqueeze(3)
    # (X / 255 - mean) / std = Y --> (Y * std + mean) * 255 = X
    images = ((image * std + mean) * 255).cpu().numpy().transpose(0, 2, 3, 1)[:, :, :, [2, 1, 0]]
    for key in featuredict.keys():
        if ('label_%d' % label) in key:
            feature = featuredict[key][0, :, :].cpu().detach().numpy()
            feature = cv2.resize(feature, images.shape[1:3])
            feature = feature / (np.max(feature) + 1e-12)
            fname = os.path.join(out_dir, 'vis_%s_%s.jpg' % (image_key, class_names[label]))
            heatmap = cv2.applyColorMap((feature * 255.).astype(np.uint8), cv2.COLORMAP_JET)
            merged = (images[0].astype(np.float) * 0.6 + heatmap.astype(np.float) * 0.4).astype(np.uint8)
            cv2.imwrite(fname, merged)

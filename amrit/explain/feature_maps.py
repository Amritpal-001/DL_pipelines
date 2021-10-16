import matplotlib.pyplot as plt

def create_forward_hook(model):
    # for layer in self.layer_names:
    t0 , t1 , t2 , t3 , t4 , t5  = [] , [] , [], [], [] , []
    # dict_keys(['conv1', 'bn1', 'act1', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'global_pool', 'fc'])

    model.conv1.register_forward_hook(lambda module, in_tensor, out_tensor: t0.append(out_tensor))
    model.layer1.register_forward_hook(lambda module, in_tensor, out_tensor: t1.append(out_tensor))
    model.layer2.register_forward_hook(lambda module, in_tensor, out_tensor: t2.append(out_tensor))
    model.layer3.register_forward_hook(lambda module, in_tensor, out_tensor: t3.append(out_tensor))
    model.layer4.register_forward_hook(lambda module, in_tensor, out_tensor: t4.append(out_tensor))
    forward_dict = {'layer1': t1,
                    'layer2': t2,
                    'layer3': t3,
                    'layer4': t4,
                    'conv1': t0, }

    return forward_dict

def create_backward_hook(model):
    # for layer in model.layer_names:
    t0, t1, t2, t3, t4, t5 = [], [], [], [], [], []
    # dict_keys(['conv1', 'bn1', 'act1', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'global_pool', 'fc'])

    model.conv1.register_backward_hook(lambda module, in_tensor, out_tensor: t0.append(out_tensor))
    model.layer1.register_backward_hook(lambda module, in_tensor, out_tensor: t1.append(out_tensor))
    model.layer2.register_backward_hook(lambda module, in_tensor, out_tensor: t2.append(out_tensor))
    model.layer3.register_backward_hook(lambda module, in_tensor, out_tensor: t3.append(out_tensor))
    model.layer4.register_backward_hook(lambda module, in_tensor, out_tensor: t4.append(out_tensor))
    backward_dict = {'layer1': t1,
                     'layer2': t2,
                     'layer3': t3,
                     'layer4': t4,
                     'conv1': t0, }
    return backward_dict

def visualize_feature_maps( features):
    fig, axs = plt.subplots(len(features) // 12, 12, figsize=(2 * 12, 2 * len(features) // 12))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(features[i].detach().abs())
        ax.axis('off')
    plt.show()
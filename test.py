import os
import argparse
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from cwmodel import cwnet
from vggmodel import *
from resnetmodel import wide_resnet
from detect_v2 import *
from tqdm import tqdm

def load_GPUS(model, model_path, **kwargs):
    state_dict = torch.load(model_path, **kwargs)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model

def build_adversarial_loader(savedpath, batch_size=1, train=False, num_workers=8, showname=False):
    dataset = AdversarialDataset(savedpath, train=train, transform=None, showname=showname)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)
    return loader

class AdversarialDataset(Dataset):
    def __init__(self, root_path: str, train: bool, transform=None, showname=False):
        self.data_root = os.path.join(root_path, 'train' if train else 'test')
        self.istrain = train
        self.transform = transform
        self.imagenames = os.listdir(self.data_root)
        self.length = len(self.imagenames)
        self.print_ = False
        self.showname = showname

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_root, self.imagenames[idx])
        data = torch.load(file_path)

        # if self.print_:
        #     print('data size is {}; {} length is {}.'. \
        #           format(data.size(), 'training set' if self.istrain else 'testing set', self.length))
        #     # self.print_ = True
        
        if self.transform:
            data = self.transform(data)

        if self.showname:
            return (data,  self.imagenames[idx])
        else:
            return data







if __name__ == "__main__":
    model_names = ['cwnet', 'wrnet', 'vgg11bn']
    attacks = ['PGD-T', 'CW-T', 'DDN-T', 'PGD-U', 'CW-U', 'DDN-U', 'FGSM', 'DFool' ]

    parser = argparse.ArgumentParser(description='TWS baseline test')
    parser.add_argument("--atk", choices=['black', 'white'], default="black", help="which type of attack you want to test, black or white")
    parser.add_argument("--dataset", choices=['fmnist', 'cifar10'], default="cifar10", help= "[cifar, fmnist] what kind of dataset you want to test")
    parser.add_argument("--data_dir", type=str, default="../wyf/SubGRED/adversarial_data/", help="datset path")
    # parser.add_argument("--data_dir", type=str, default="../fmnist/adversarial_data/", help="datset path")
    parser.add_argument("--model_dir", type=str, default='../wyf/SubGRED/classifier_pth/', help="path to the model you want to test")
    # parser.add_argument("--model_dir", type=str, default='../fmnist/classifier_pth/', help="path to the model you want to test")
    parser.add_argument("--arch", choices=model_names, default='vgg11bn', help="the model arch from which the adversarial examples are generated")
    # parser.add_argument('atk', choices=attacks, required=True, help="the attack you want to test against")
    args = parser.parse_args()

    if args.atk == "black":
        adv_src_model = 'wrnet' if args.dataset == "cifar" else "cwnet"
    elif args.atk == "white":
        adv_src_model = args.arch
    else:
        raise Exception("Unknown type of test")

    model_root = os.path.join(args.model_dir, 'classifier_{}_{}_best.pth'.format(args.arch, args.dataset))
    if args.arch == "cwnet" or args.arch == "vgg11bn":
        inch = 3 if args.dataset == 'cifar10' else 1
        model = eval(args.arch)(num_classes=10, inchannels=inch)
        model.load_state_dict(torch.load(model_root))
    elif args.arch == "wrnet":
        model = wide_resnet(depth=28, num_classes=10, widen_factor=10) # only for cifar10
        model = load_GPUS(classifier, model_root)
    else:
        raise Exception("Unknown model type")

    model.to(torch.device('cuda'))
    model.eval()

    # record the detail of this exp in str?
    exp_name = '{}_{}'.format(args.dataset, args.atk)

    # Some parameters
    n_radius = 0.01 #specifies the noise radius in l1 norm detection, used in C1 to measure robustness.
    targeted_lr = 0.0005 #specifies the learning rate for targeted attack detection criterion, used in C2t
    t_radius = 0.5 #specifies the radius of targeted attack detection criterion, used in C2t
    u_radius = 0.5 #specifies the radius of untargeted attack detection criterion, used in C2u
    untargeted_lr = 1 #specifies the learning rate for untargeted attack detection criterion, used in C2u
    # Clean data loader
    adv_root = os.path.join(args.data_dir, '{}_{}_{}'.format(args.dataset, adv_src_model, "Clean"))
    test_loader = build_adversarial_loader(adv_root, train=False, showname=True)
    # Calculate the fpr
    target_1 = l1_vals(model, test_loader, "Clean", n_radius)
    target_2 = targeted_vals(model, test_loader, "Clean", targeted_lr, t_radius)
    target_3 = untargeted_vals(model, test_loader, "Clean", untargeted_lr, u_radius)
    p_1 = target_1.copy()
    p_2 = target_2.copy()
    p_3 = target_3.copy()
    p_1.sort()
    p_2.sort()
    p_3.sort()
    fpr = np.zeros(len(p_1)*len(p_2)*len(p_3)+1)
    for ix_1 in tqdm(range(0,len(p_1)), desc=exp_name+'_clean'):
        for ix_2 in range(0,len(p_2)):
            for ix_3 in range(0,len(p_3)):
                fpr[len(p_2)*len(p_3)*ix_1+len(p_3)*ix_2+ix_3] = len(target_1[np.logical_or(np.logical_or(target_1>p_1[ix_1],target_2>p_2[ix_2]),target_3>p_3[ix_3])])*1.0/len(target_1)
    fpr[-1] = len(target_1[np.logical_or(np.logical_or(target_1>=p_1[-1],target_2>=p_2[-1]),target_3>=p_3[-1])])*1.0/len(target_1)
    
    # save target 1,2,3 and fpr list (all np array)
    np.save(exp_name+'clean_val1', target_1)
    np.save(exp_name+'clean_val2', target_2)
    np.save(exp_name+'clean_val3', target_3)
    np.save(exp_name+'fpr', fpr)

    # Calculate tpr for different attacks
    # the test loop over all the attacks in the list
    for i, attack in enumerate(attacks):
        adv_root = os.path.join(args.data_dir, '{}_{}_{}'.format(args.dataset, adv_src_model, attack))
        test_loader = build_adversarial_loader(adv_root, train=False, showname=True)
        
        a_target_1 = l1_vals(model, test_loader, attacks[i], n_radius)[::-1]
        a_target_2 = targeted_vals(model, test_loader, attacks[i], targeted_lr, t_radius)[::-1]
        a_target_3 = untargeted_vals(model, test_loader, attacks[i], untargeted_lr, u_radius)[::-1]

        results = {}
        auc_list = np.zeros(201)
        counter = np.zeros(201)
        tpr = np.zeros(len(p_1)*len(p_2)*len(p_3)+1)
        for ix_1 in tqdm(range(0,len(p_1)), desc=exp_name+"_"+attacks[i]):
            for ix_2 in range(0,len(p_2)):
                for ix_3 in range(0,len(p_3)):
                    tpr[len(p_2)*len(p_3)*ix_1+len(p_3)*ix_2+ix_3] = len(a_target_1[np.logical_or(np.logical_or(a_target_1>p_1[ix_1],a_target_2>p_2[ix_2]),a_target_3>p_3[ix_3])])*1.0/len(a_target_1)

                    # accumulate this tpr to the corresponding auc list
                    index = int(np.round(fpr[len(p_2)*len(p_3)*ix_1+len(p_3)*ix_2+ix_3]*200))
                    counter[index] += 1
                    auc_list[index] += tpr[len(p_2)*len(p_3)*ix_1+len(p_3)*ix_2+ix_3]
                    # if fpr[len(p_2)*len(p_3)*ix_1+len(p_3)*ix_2+ix_3] <= target_fpr and fpr[len(p_2)*len(p_3)*ix_1+len(p_3)*ix_2+ix_3] > target_fpr-0.01:
                    #     suitable_pairs.append((p_1[ix_1],p_2[ix_2],p_3[ix_3]))
                    #     tprs.append(tpr[len(p_2)*len(p_3)*ix_1+len(p_3)*ix_2+ix_3])

        print("There are total of %d groups in %s has no data"%(sum(counter == 0)), attacks[i])
        counter[counter == 0] += 1 
        results[attacks[i]] = sum(auc_list / counter)*0.005
        tpr[-1] = len(a_target_1[np.logical_or(np.logical_or(a_target_1>=p_1[-1],a_target_2>=p_2[-1]),a_target_3>=p_3[-1])])*1.0/len(a_target_1)
        # save attacks' corresponding tpr list
        np.save(exp_name+'_'+attacks[i]+'_tpr', tpr)
        print("Results for %s is "%exp_name+attacks[i])
        print(results)





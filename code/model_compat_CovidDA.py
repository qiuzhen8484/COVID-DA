import torch.nn as nn
from functions import ReverseLayerF
from net.resnet import resnet18
import torch
import torch.nn.functional as F


class COVID_DA_with_resnet18(nn.Module):
    def __init__(self, args):
        super(COVID_DA_with_resnet18, self).__init__()
        if args.image_size[0] == 224:
            self.shared_encoder_conv = resnet18(pretrained=True)
        else:
            self.shared_encoder_conv = resnet18(pretrained=False)
        # classify 2 classes(main task)
        # domain-shared classifier
        self.shared_classifier = Classifier()
        # source-specific classifier
        self.source_specific_classifier = Classifier()
        # target specific classifier
        self.target_specific_classifier = Classifier()

        # construct D1, D2 discriminator
        self.discriminator1 = Discriminator1() 
        self.discriminator2 = Discriminator2() 
  

    def forward(self, source_data, target_data, p=0.0):
        # input: [source data, target data]
        # output: [source feature, target feature, all feature, domain_label, shared class label, domain_label_joint, \
        # private_source_label, final source class label, private_target_label, final target class label]
        result = []
        # Feature Extractor
        _, source_GAP = self.shared_encoder_conv(source_data)
        _, target_GAP = self.shared_encoder_conv(target_data)
 
        source_flatten = source_GAP.view(-1, 512)
        target_flatten = target_GAP.view(-1, 512)
        all_features = torch.cat([source_flatten, target_flatten], 0)
        result.append(source_flatten)
        result.append(target_flatten)        
        result.append(all_features)

        # data number
        source_sample_num = source_flatten.size(0)

        # Discriminator 1
        reversed_G_flatten = ReverseLayerF.apply(all_features, p)
        domain_label = self.discriminator1(reversed_G_flatten)
        result.append(domain_label)

        # Shared classifier 
        shared_class_label = self.shared_classifier(all_features)
        result.append(shared_class_label)

        shared_source_label = shared_class_label[:source_sample_num,:]
        shared_target_label = shared_class_label[source_sample_num:,:]

        # Discriminator 2
        # note that: we do not train the encoder here
        reversed_C_flatten = ReverseLayerF.apply(shared_class_label, p)
        domain_label_joint = self.discriminator2(reversed_G_flatten, reversed_C_flatten)
        result.append(domain_label_joint)

        # source prediction
        private_source_label = self.source_specific_classifier(source_flatten)
        # final source prediction 
        final_source_label = (F.softmax(private_source_label, dim=1) + F.softmax(shared_source_label, dim=1))/2
        result.append(private_source_label)
        result.append(final_source_label)

        # target prediction
        private_target_label = self.target_specific_classifier(target_flatten)
        # final target prediction 
        final_target_label = (F.softmax(private_target_label, dim=1) + F.softmax(shared_target_label, dim=1))/2
        result.append(private_target_label)
        result.append(final_target_label)

        return result


class Classifier(nn.Module):
    def __init__(self, n_class=2):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential()
        self.fc.add_module('fc', nn.Linear(in_features=512, out_features=n_class))

    def forward(self, input_data):
        class_label = self.fc(input_data)
        return class_label


class Discriminator1(nn.Module):
    ## feature adaptation
    def __init__(self):
        super(Discriminator1, self).__init__()
        self.shared_encoder_pred_domain = nn.Sequential()
        self.shared_encoder_pred_domain.add_module('d_se6', nn.Linear(in_features=512, out_features=1024))
        self.shared_encoder_pred_domain.add_module('relu_se7', nn.LeakyReLU(0.01))

        # classify two domain
        self.shared_encoder_pred_domain.add_module('d_se8', nn.Linear(in_features=1024, out_features=1))

    def forward(self, input_data):
        domain_label = self.shared_encoder_pred_domain(input_data)
   
        return domain_label


class Discriminator2(nn.Module):
    """
    Discriminator 2 with leaky relu
    input:
          input_feature: output of domain-shared  feature extractor
          input_prediction: output of domain-shared classifier
    output:
          domain_label_joint: predicted domain label
    """
    def __init__(self):
        super(Discriminator2, self).__init__()
        self.shared_encoder_pred_fc = nn.Sequential()
        self.shared_encoder_pred_fc.add_module('d_se6', nn.Linear(in_features=512, out_features=16))
        self.shared_encoder_pred_fc.add_module('relu_se7', nn.LeakyReLU(0.01))

        self.shared_classifier_pred_fc = nn.Sequential()
        self.shared_classifier_pred_fc.add_module('d_se8', nn.Linear(in_features=2, out_features=16))
        self.shared_classifier_pred_fc.add_module('relu_se9', nn.LeakyReLU(0.01))

        self.discrminator_pre_domain = nn.Sequential()
        self.discrminator_pre_domain.add_module('d_se10', nn.Linear(in_features=32, out_features=32))
        self.discrminator_pre_domain.add_module('relu_se11', nn.LeakyReLU(0.01))
        # classify two domain
        self.discrminator_pre_domain.add_module('fc_se12', nn.Linear(in_features=32, out_features=1))

    def forward(self, input_feature, input_prediction):
        feature = input_feature
        prediction = input_prediction
        shared_encoder_pred_domain = self.shared_encoder_pred_fc(feature)
        shared_classifier_pred_domain = self.shared_classifier_pred_fc(prediction)
        discrminator_pre_domain_input = torch.cat([shared_encoder_pred_domain, shared_classifier_pred_domain], 1)
        domain_label_joint = self.discrminator_pre_domain(discrminator_pre_domain_input)
       
        return domain_label_joint

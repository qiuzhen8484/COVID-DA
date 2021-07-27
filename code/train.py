import torch.optim as optim
import torch.utils.data
from torchvision import transforms
import numpy as np
from torch.autograd import Variable
from loss_define import entropy_loss, focal_loss, ls_distance
from dataset import covidDataset_target
from utils import analyse


def val(my_net, logger, epoch, name, args=None):
    batch_size = args.batchsize
    image_size = args.image_size

    ###################
    # load data       #
    ###################

    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5002, 0.5002, 0.5002), (0.0893, 0.0893, 0.0893)),  # grayscale mean/std
    ])

    if name == 'val':
        test_dataset = covidDataset_target(args.path_target, args.label_file_target, train=False, transform=img_transform)
        dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    else:
        print('error dataset name')

    my_net.eval()

    if args.cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0
    prob_list = []
    gt_list = []

    while i < len_dataloader:

        data_input = data_iter.next()
        img, label = data_input
        gt_list.append(label.numpy())

        batch_size = len(label)

        input_img = torch.FloatTensor(batch_size, 3, image_size[0], image_size[1])
        class_label = torch.LongTensor(batch_size)

        if args.cuda:
            img = img.cuda()
            label = label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        input_img.resize_as_(input_img).copy_(img)
        class_label.resize_as_(label).copy_(label)
        inputv_img = Variable(input_img)
        classv_label = Variable(class_label)

        result = my_net(inputv_img, inputv_img)
        pred = result[-1].data.max(1, keepdim=True)[1]
        output_prob = result[-1].data
        prob_list.append(output_prob[:, 1].detach().cpu().numpy())
        n_correct += pred.eq(classv_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = np.float(n_correct * 1.0) / np.float(n_total)
    prob_list = np.concatenate(prob_list)
    gt_list = np.concatenate(gt_list)

    F1 = analyse(gt_list, prob_list, prob=True, logger=logger)
    logger.info('epoch: %d, accuracy of the %s dataset: %f' % (epoch, name, accu))
    logger.info('epoch: %d, F1 of the %s dataset: %f' % (epoch, name, F1))
    return accu, F1


def train_covid_da(my_net, dataloader_source, dataloader_target_unlabeled, dataloader_target_labeled, args, logger):

    def exp_lr_scheduler(optimizer, step, init_lr=args.lr, lr_decay_step=args.lr_decay_step, step_decay_weight=args.step_decay_weight):

        # Decay learning rate by a factor of step_decay_weight every lr_decay_step
        current_lr = init_lr * (step_decay_weight ** (step / lr_decay_step))

        if step % lr_decay_step == 0:
            print('learning rate is set to %f' % current_lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        return optimizer

    #####################
    # setup optimizer   #
    #####################

    # classification loss optimizer
    optimizer = optim.SGD(
        list(my_net.module.shared_encoder_conv.parameters()) + list(
            my_net.module.shared_classifier.parameters()) + list(my_net.module.source_specific_classifier.parameters())
        + list(my_net.module.target_specific_classifier.parameters()),
        lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay)

    # D2 optimizer
    optimizer_D2 = optim.SGD(
        list(my_net.module.shared_classifier.parameters()) + list(my_net.module.source_specific_classifier.parameters())
        + list(my_net.module.target_specific_classifier.parameters()) + list(my_net.module.discriminator2.parameters()),
        lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay)

    # D1 optimizer
    optimizer_D1 = optim.SGD(
        list(my_net.module.shared_encoder_conv.parameters()) + list(my_net.module.discriminator1.parameters()),
        lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay)

    loss_classification = focal_loss(alpha=0.25)
    loss_entropy = entropy_loss()
    loss_similarity = torch.nn.CosineSimilarity()

    if args.cuda:
        my_net = my_net.cuda()
        loss_classification = loss_classification.cuda()
        loss_entropy = loss_entropy.cuda()
        loss_similarity = loss_similarity.cuda()

    for p in my_net.parameters():
        p.requires_grad = True

    #############################
    # training network          #
    #############################

    len_dataloader = min(len(dataloader_source), len(dataloader_target_unlabeled))

    current_step = 0
    for epoch in range(args.epoch):
        my_net.train()

        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target_unlabeled)
        data_target_semi_iter = iter(dataloader_target_labeled)
        num_iter = len(dataloader_source)

        for i in range(1, num_iter + 1):

            ###################################
            #      load the training data     #
            ###################################

            source_data, source_label = data_source_iter.next()
            target_data, target_label = data_target_iter.next()
            target_data_semi, target_label_semi = data_target_semi_iter.next()

            if i % len(dataloader_target_unlabeled) == 0:
                data_target_iter = iter(dataloader_target_unlabeled)

            if i % len(dataloader_target_labeled) == 0:
                data_target_semi_iter = iter(dataloader_target_labeled)

            # transform data
            src_label_dm = torch.ones(source_label.size()).long()
            tgt_label_dm = torch.zeros(target_label.size()).long()
            tgt_label_dm_semi = torch.zeros(target_label_semi.size()).long()

            if args.cuda:
                src_data, src_label_cl, src_label_dm = source_data.cuda(), source_label.cuda(), src_label_dm.cuda()
                tgt_data, tgt_label_cl, tgt_label_dm = target_data.cuda(), target_label.cuda(), tgt_label_dm.cuda()
                tgt_data_semi, tgt_label_cl_semi, tgt_label_dm_semi = target_data_semi.cuda(), target_label_semi.cuda(), tgt_label_dm_semi.cuda()

            src_data, src_label_cl, src_label_dm = Variable(src_data), Variable(src_label_cl), Variable(src_label_dm)
            tgt_data, tgt_label_cl, tgt_label_dm = Variable(tgt_data), Variable(tgt_label_cl), Variable(tgt_label_dm)
            tgt_data_semi, tgt_label_cl_semi, tgt_label_dm_semi = Variable(tgt_data_semi), Variable(
                tgt_label_cl_semi), Variable(tgt_label_dm_semi)

            # mark down the batch size of source data and target data
            src_num = src_label_cl.size(0)
            tar_label_num = tgt_label_cl_semi.size(0)

            p = float(i + epoch * len_dataloader) / args.epoch / len_dataloader
            p = 2. / (1. + np.exp(-10 * p)) - 1

            ###################################
            #       main task training        #
            ###################################
            optimizer.zero_grad()
            loss = 0

            result = my_net(src_data, torch.cat([tgt_data_semi, tgt_data], 0), p=p)
            src_class_pre, tgt_class_pre = result[7], result[9]

            # optimization of classification loss
            source_classification = loss_classification(src_class_pre, src_label_cl)
            tgt_semi_classification = loss_classification(tgt_class_pre[:tar_label_num, :], tgt_label_cl_semi)
            loss += source_classification + tgt_semi_classification

            all_entropy = args.gamma_weight * loss_entropy(torch.cat([src_class_pre, tgt_class_pre], 0))
            loss += all_entropy

            loss.backward()
            optimizer = exp_lr_scheduler(optimizer=optimizer, step=current_step)
            optimizer.step()

            ###################################
            #           D1 training           #
            ###################################
            optimizer_D1.zero_grad()
            loss = 0

            result = my_net(src_data, torch.cat([tgt_data_semi, tgt_data], 0), p=p)
            encoder_fea = result[3]

            # optimization of domain loss 1 for discriminator 1
            source_dann = args.alpha_weight * ls_distance(encoder_fea[:src_num], flag='source')
            tgt_dann = args.alpha_weight * ls_distance(encoder_fea[src_num:], flag='target')
            loss += source_dann + tgt_dann

            loss.backward()
            optimizer_D1 = exp_lr_scheduler(optimizer=optimizer_D1, step=current_step)
            optimizer_D1.step()

            ###################################
            #           D2 training           #
            ###################################
            optimizer_D2.zero_grad()
            loss = 0

            result = my_net(src_data, torch.cat([tgt_data_semi, tgt_data], 0), p=p)
            D2_dm_fea = result[5]

            # optimization of domain loss 2 and diversity loss
            source_dann = args.alpha_weight * ls_distance(D2_dm_fea[:src_num], flag='source')
            tgt_dann = args.alpha_weight * ls_distance(D2_dm_fea[src_num:], flag='target')
            loss += source_dann + tgt_dann

            share_label_pre, src_private_pre, tgt_private_pre = result[4], result[6], result[8]
            src_difference = args.beta_weight * loss_similarity(share_label_pre[:src_num], src_private_pre)
            tgt_difference = args.beta_weight * loss_similarity(share_label_pre[src_num:], tgt_private_pre)

            diversity = torch.mean(src_difference) + torch.mean(tgt_difference)

            loss += diversity
            loss.backward()
            optimizer_D2 = exp_lr_scheduler(optimizer=optimizer_D2, step=current_step)
            optimizer_D2.step()
            current_step += 1

        logger.info(
            'source_classification: %f, source_dann: %f, target_dann: %f, all_entropy: %f, diversity: %f'
            % (source_classification.data.cpu().numpy(), source_dann.data.cpu().numpy(), tgt_dann.data.cpu().numpy(),
               all_entropy.data.cpu().numpy(), diversity.data.cpu().numpy()))
        logger.info('loss: %f' % loss.data)

        torch.save(my_net, args.model_root + '/' + args.time_stamp_launch + '-COVID-DA_last' + '.pkl')
        accu, F1 = val(my_net, logger, epoch=epoch, name='val', args=args)
        if F1 >= args.max_F1:
            logger.info('saving the best model!')
            torch.save(my_net, args.model_root + '/' + args.time_stamp_launch + '-COVID-DA_best' + '.pkl')
            args.max_F1 = F1

        logger.info('acc is : %.04f, F1 is : %.04f, best F1 is : %.04f' % (accu, F1, args.max_F1))
        logger.info('================================================')

    logger.info('done')

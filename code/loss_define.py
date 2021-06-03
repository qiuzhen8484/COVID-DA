import torch
import torch.nn as nn
import torch.nn.functional as F


class entropy_loss(nn.Module):
    def __init__(self):
        super(entropy_loss, self).__init__()

    def forward(self, logits):
        y_pred = F.softmax(logits, dim=-1)
        size = logits.size(0)
        if size == 0:
            loss = 0.0
        else:
            loss = torch.sum(-y_pred * torch.log(y_pred), dim=1)
        return torch.mean(loss)


def js_div(p_output, q_output, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output)
        q_output = F.softmax(q_output)
    log_mean_output = ((p_output + q_output )/2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2


class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=2, size_average=True):
        """
        focal_loss
        :param alpha:   default: 0.25
        :param gamma:   default: 2
        :param num_classes:     the number of classes
        :param size_average:    default: True
        """

        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha) == num_classes  # alpha can be a list to assign weights for each class,size:[num_classes]
            print("Focal_loss alpha = {}".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            print(" --- Focal_loss alpha = {}  --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss calculation
        :param preds:   the predicted probability. size:[B,N,C] or [B,C]
        :param labels:  the ground truth. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = preds
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def ls_distance(logits, flag='source'):
    # least square distance for domain loss
    if flag == 'source':
        domain_loss = torch.mean((logits) ** 2)
    else:
        domain_loss = torch.mean((logits - 1) ** 2)
    return domain_loss


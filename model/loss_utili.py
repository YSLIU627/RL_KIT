import math 

def huber_loss(self, input, target, beta=1, size_average=True):
        """
        a method of  defining loss which increase the robustness of computing on discrete data
        """
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        if size_average:
            return loss.mean()
        return loss.sum()
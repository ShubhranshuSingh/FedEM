import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnersEnsemble(object):
    """
    Iterable Ensemble of Learners.

    Attributes
    ----------
    learners
    learners_weights
    model_dim
    is_binary_classification
    device
    metric

    Methods
    ----------
    __init__
    __iter__
    __len__
    compute_gradients_and_loss
    optimizer_step
    fit_epochs
    evaluate
    gather_losses
    free_memory
    free_gradients

    """
    def __init__(self, learners, learners_weights, optimizer, lr_scheduler = None):
        self.learners = learners
        self.learners_weights = learners_weights
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.model_dim = self.learners[0].model_dim
        self.is_binary_classification = self.learners[0].is_binary_classification
        self.device = self.learners[0].device
        self.metric = self.learners[0].metric
        self.log_offset = 1e-20
        self.det_offset = 1e-6
        
        self.alpha = 2.0
        self.beta = 0.5

    def optimizer_step(self):
        """
        perform one optimizer step, requires the gradients to be already computed
        """
        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()

    def compute_gradients_and_loss(self, batch, weights=None):
        """
        compute the gradients and loss over one batch.

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss

        """
        self.optimizer.zero_grad()
        total_loss = 0
        losses = []
        
        x, y, indices = batch
        x = x.to(self.device).type(torch.float32)
        y = y.to(self.device)
        
        # One hot
        num_classes = list(self.learners[0].model.parameters())[-1].shape[0]
        y_true = torch.zeros(x.size(0), num_classes).to(self.device)
        y_true.scatter_(1, y.view(-1,1), 1)
        mask_non_y_pred = []
        ensemble_probs = 0
        
        for learner_id, learner in enumerate(self.learners):
            loss, y_pred = learner.compute_gradients_and_loss(batch, weights=weights[learner_id])
  
            total_loss += loss*self.learners_weights[learner_id] # all models in ensemble have different weight
            
            y_pred = F.softmax(y_pred, dim=-1)
            ensemble_probs += y_pred*self.learners_weights[learner_id] # all models in ensemble have different weight
            
            bool_R_y_true = torch.eq(torch.ones_like(y_true) - y_true, torch.ones_like(y_true))
            mask_non_y_pred.append(torch.masked_select(y_pred*self.learners_weights[learner_id], bool_R_y_true).reshape(-1, num_classes-1))
            
            losses.append(loss.detach())
            
        ensemble_entropy = torch.sum(-torch.mul(ensemble_probs, torch.log(ensemble_probs + self.log_offset)), dim=-1).mean()

        mask_non_y_pred = torch.stack(mask_non_y_pred, dim=1)
        assert mask_non_y_pred.shape == (x.size(0), len(self.learners), num_classes-1)
        
        mask_non_y_pred = mask_non_y_pred / torch.norm(mask_non_y_pred, p=2, dim=-1, keepdim=True) 
        matrix = torch.matmul(mask_non_y_pred, mask_non_y_pred.permute(0, 2, 1))
        log_det = torch.logdet(matrix+self.det_offset*torch.eye(len(self.learners), device=matrix.device).unsqueeze(0)).mean() 
        
        total_loss *= len(self.learners)
        total_loss -= self.alpha * ensemble_entropy + self.beta * log_det
        
        total_loss.backward()

        return losses

    def fit_batch(self, batch, weights):
        """
        updates learners using  one batch.

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            client_updates (np.array, shape=(n_learners, model_dim)): the difference between the old parameter
            and the updated parameters for each learner in the ensemble.

        """
        client_updates = torch.zeros(len(self.learners), self.model_dim)
        
        for learner_id, learner in enumerate(self.learners):
            client_updates[learner_id] = learner.get_param_tensor()
            
            
        self.compute_gradients_and_loss(batch, weights)
        self.optimizer_step()
        
        for learner_id, learner in enumerate(self.learners):
            client_updates[learner_id] = learner.get_param_tensor() - client_updates[learner_id].to(self.device)

        return client_updates.cpu().numpy()

    def fit_epochs(self, iterator, n_epochs, weights=None):
        """
        perform multiple training epochs, updating each learner in the ensemble

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param n_epochs: number of epochs
        :type n_epochs: int
        :param weights: tensor of shape (n_learners, len(iterator)), holding the weight of each sample in iterator
                        for each learner ins ensemble_learners
        :type weights: torch.tensor or None
        :return:
            client_updates (np.array, shape=(n_learners, model_dim)): the difference between the old parameter
            and the updated parameters for each learner in the ensemble.

        """
        
        client_updates = torch.zeros(len(self.learners), self.model_dim)
        
        for learner_id, learner in enumerate(self.learners):
            client_updates[learner_id] = learner.get_param_tensor()
            
        for i in range(n_epochs):
            for batch in iterator:
                self.compute_gradients_and_loss(batch,weights)
                self.optimizer.step()
            if self.lr_scheduler:
                self.lr_scheduler.step()
                
        for learner_id, learner in enumerate(self.learners):
            client_updates[learner_id] = learner.get_param_tensor() - client_updates[learner_id].to(self.device)

        return client_updates.cpu().numpy()

    def evaluate_iterator(self, iterator):
        """
        Evaluate a ensemble of learners on iterator.

        :param iterator: yields x, y, indices
        :type iterator: torch.utils.data.DataLoader
        :return: global_loss, global_acc

        """
        if self.is_binary_classification:
            criterion = nn.BCELoss(reduction="none")
        else:
            criterion = nn.NLLLoss(reduction="none")

        for learner in self.learners:
            learner.model.eval()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        with torch.no_grad():
            for (x, y, _) in iterator:
                x = x.to(self.device).type(torch.float32)
                y = y.to(self.device)
                n_samples += y.size(0)

                y_pred = 0.
                for learner_id, learner in enumerate(self.learners):
                    if self.is_binary_classification:
                        y_pred += self.learners_weights[learner_id] * torch.sigmoid(learner.model(x))
                    else:
                        y_pred += self.learners_weights[learner_id] * F.softmax(learner.model(x), dim=1)

                y_pred = torch.clamp(y_pred, min=0., max=1.)

                if self.is_binary_classification:
                    y = y.type(torch.float32).unsqueeze(1)
                    global_loss += criterion(y_pred, y).sum().item()
                    y_pred = torch.logit(y_pred, eps=1e-10)
                else:
                    global_loss += criterion(torch.log(y_pred), y).sum().item()

                global_metric += self.metric(y_pred, y).item()

            return global_loss / n_samples, global_metric / n_samples

    def gather_losses(self, iterator):
        """
        gathers losses for all sample in iterator for each learner in ensemble

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            tensor (n_learners, n_samples) with losses of all elements of the iterator.dataset

        """
        n_samples = len(iterator.dataset)
        all_losses = torch.zeros(len(self.learners), n_samples)
        for learner_id, learner in enumerate(self.learners):
            all_losses[learner_id] = learner.gather_losses(iterator)

        return all_losses

    def free_memory(self):
        """
        free_memory: free the memory allocated by the model weights

        """
        for learner in self.learners:
            learner.free_memory()

    def free_gradients(self):
        """
        free memory allocated by gradients

        """
        for learner in self.learners:
            learner.free_gradients()

    def __iter__(self):
        return LearnersEnsembleIterator(self)

    def __len__(self):
        return len(self.learners)

    def __getitem__(self, idx):
        return self.learners[idx]


class LanguageModelingLearnersEnsemble(LearnersEnsemble):
    def evaluate_iterator(self, iterator):
        """
        Evaluate a ensemble of learners on iterator.

        :param iterator: yields x, y, indices
        :type iterator: torch.utils.data.DataLoader
        :return: global_loss, global_acc

        """
        criterion = nn.NLLLoss(reduction="none")

        for learner in self.learners:
            learner.model.eval()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        with torch.no_grad():
            for (x, y, _) in iterator:
                x = x.to(self.device)
                y = y.to(self.device)
                n_samples += y.size(0)
                chunk_len = y.size(1)

                y_pred = 0.
                for learner_id, learner in enumerate(self.learners):
                    y_pred += self.learners_weights[learner_id] * F.softmax(learner.model(x), dim=1)

                y_pred = torch.clamp(y_pred, min=0., max=1.)

                global_loss += criterion(torch.log(y_pred), y).sum().item() / chunk_len
                global_metric += self.metric(y_pred, y).item() / chunk_len

            return global_loss / n_samples, global_metric / n_samples


class LearnersEnsembleIterator(object):
    """
    LearnersEnsemble iterator class

    Attributes
    ----------
    _learners_ensemble
    _index

    Methods
    ----------
    __init__
    __next__

    """
    def __init__(self, learners_ensemble):
        self._learners_ensemble = learners_ensemble.learners
        self._index = 0

    def __next__(self):
        while self._index < len(self._learners_ensemble):
            result = self._learners_ensemble[self._index]
            self._index += 1

            return result

        raise StopIteration

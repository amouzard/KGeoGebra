#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset

from collections import defaultdict


class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.pos = torch.tensor([1])
        self.neg = torch.tensor([1])
        self.none = torch.tensor([0])
        self.mrr = 1
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.pi = torch.pi

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        ###########################################################################
        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim
        if model_name == "EllipsE":
            self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        if model_name == "EllipsEs":
            self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
            self.relation_dim = hidden_dim * 3 if double_relation_embedding else hidden_dim
        if model_name in ["ButtErflies", "ComplEx"]:
            self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
            self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim
        if model_name == "ButtErfly":
            self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if model_name in ['EllipsE', 'EllipsEs', 'ButtErflies', 'ButtErfly']:
            self.ellipse = nn.Parameter(torch.rand((2 * self.hidden_dim)), requires_grad=True)

        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['EllipsE', 'EllipsEs', 'ButtErflies', 'ButtErfly', 'RotatE', 'ComplEx', 'TransE', 'DistMult']:
            raise ValueError('model %s not supported' % model_name)


    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'EllipsE': self.EllipsE,
            'EllipsEs': self.EllipsEs,
            'ButtErflies': self.ButtErflies,
            'ButtErfly': self.ButtErfly,
            'RotatE': self.RotatE,
            'ComplEx': self.ComplEx,
            'TransE': self.TransE,
            'DistMult': self.DistMult,
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score
    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - (score.sum(dim=2) + self.mrr)
        return score

    def EllipsE(self, head, rel, tail, mode):
        MIN = 1e-10
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = rel/(self.embedding_range.item()/self.pi)
        a, b = self.ellipse[:self.hidden_dim], self.ellipse[self.hidden_dim:]
        a, b = a.clamp_min(MIN), b.clamp_min(MIN)
        re_relation = a * torch.cos(phase_relation)
        im_relation = b * torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation - re_tail
        im_score = re_head * im_relation + im_head * re_relation - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0, p=2)
        score = self.gamma.item() - score.sum(dim=2)
        return score

    def EllipsEs(self, head, rel1, tail, mode):
        MIN = 1e-10
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        a, b, rel = torch.chunk(rel1, 3, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = rel/(self.embedding_range.item()/self.pi)
        a = torch.abs(a).clamp_min(MIN)
        b = torch.abs(b).clamp_min(MIN)

        re_relation = a * torch.cos(phase_relation)
        im_relation = b * torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation - re_tail
        im_score = re_head * im_relation + im_head * re_relation - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0, p=2)
        score = self.gamma.item() - score.sum(dim=2)
        return score

    def ButtErfly(self, head, rel, tail, mode):
        # Embeds entities in standard complex space and relation on butterflies. From the relation angle,
        # the butterfly polar equation are used to find relation radius.
        # Relations become standard complex numbers and multiply the heads.
        MIN = 1e-10
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        theta = rel
        # Make phases of relations uniformly distributed in [-pi, pi]
        theta = 12*self.pi + 12*theta/(self.embedding_range.item()/self.pi)
        radius = torch.exp(torch.sin(theta)) - 2*torch.cos(4*theta) + torch.sin((2*theta - self.pi)/24.)**5

        re_relation = radius * torch.cos(theta)
        im_relation = radius * torch.sin(theta)
        re_score = (re_head * re_relation - im_head * im_relation) - re_tail
        im_score = (re_head * im_relation + im_head * re_relation) - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0, p=2)
        score = self.gamma.item() - score.sum(dim=2)
        return score

    def ButtErflies(self, head, rel, tail, mode):
        # Embeds entities in standard complex space and relation on butterflies.
        MIN = 1e-10
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        theta, bias = torch.chunk(rel, 2, dim=2)
        bias = torch.abs(bias)
        # Make phases of relations uniformly distributed in [-pi, pi]
        theta = 12*self.pi + 12*theta/(self.embedding_range.item()/self.pi)
        radius = torch.exp(torch.sin(theta)) - 2*torch.cos(4*theta) + torch.sin((2*theta - self.pi)/24.)**5

        radius = radius * bias
        re_relation = radius * torch.cos(theta)
        im_relation = radius * torch.sin(theta)
        re_score = (re_head * re_relation - im_head * im_relation) - re_tail
        im_score = (re_head * im_relation + im_head * re_relation) - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0, p=2)
        score = self.gamma.item() - score.sum(dim=2)
        return score

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propagation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        # true_in = [(negative_sample[i] == v[2]).sum().item() for i, v in enumerate(positive_sample)]
        # print(mode, torch.sum(torch.tensor(true_in)))
        # exit()
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)
        # negative_score is a bs * ng array which contains the score of all -ve_triple obtained from the -ve entities.
        # print(negative_score.shape,'\n', negative_score,'\n', mode)
        ## Start computing the LOSS for each +ve triple
        if args.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)
        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                    model.entity_embedding.norm(p=3) ** 3 +
                    model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()
        if args.countries:
            # Countries S* datasets are evaluated on AUC-PR
            # Process test data for AUC-PR evaluation
            sample = list()
            y_true = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            # average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}

        else:
            # Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            # Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'head-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'tail-batch'
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num // 2),
                collate_fn=TestDataset.collate_fn
            )
            if args.save_ranks:
                dt=pd.DataFrame(data=test_triples, columns=['h', 'r', 't'])
                triple_frame = pd.DataFrame(data=0, columns=['h', 'r', 't', 'head-batch', 'tail-batch', 'ranks'], index=range(len(test_triples)))
                triple_frame.loc[:, ['h', 'r', 't']] = [[h,r,t] for h,r,t in test_triples]


            test_dataset_list = [test_dataloader_head, test_dataloader_tail]

            logs = []
            logs_rel = defaultdict(list)  # logs for every relation
            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])
            file = open('scoring_short.txt', 'w')
            with torch.no_grad():
                for data_index, test_dataset in enumerate(test_dataset_list):
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda() # shape bs_test * 3. bs_test = 8, 16
                            negative_sample = negative_sample.cuda() # shape bs_test * ne
                            filter_bias = filter_bias.cuda() # shape bs_test * ne
                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), mode) # shape bs_test * ne
                        score += filter_bias

                        # Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim=1, descending=True) # shape bs_test * ne

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)


                        for i in range(batch_size):
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()

                            rel = positive_sample[i][1].item()

                            # ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()

                            triple_ind = tuple([int(v) for v in positive_sample[i].tolist()])
                            assert triple_ind in test_triples
                            if args.save_ranks:
                                conds = (triple_frame.h == triple_ind[0]) & (triple_frame.r == triple_ind[1]) & (triple_frame.t == triple_ind[2])
                                triple_frame.loc[:, mode][conds] = ranking

                            log = {
                                '******* Model '+args.model+' **** ': 1,
                                'MRR': 1.0 / ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            }

                            if args.model in ['EllipsE']:
                                log.update(
                                    {
                                        args.model + ' a mean': model.ellipse[:model.hidden_dim].mean().item(),
                                        args.model + ' a std': model.ellipse[:model.hidden_dim].std().item(),
                                        args.model + ' a min': model.ellipse[:model.hidden_dim].min().item(),
                                        args.model + ' a max': model.ellipse[:model.hidden_dim].max().item(),
                                        args.model + ' b mean': model.ellipse[model.hidden_dim:].mean().item(),
                                        args.model + ' b std': model.ellipse[model.hidden_dim:].std().item(),
                                        args.model + ' b min': model.ellipse[model.hidden_dim:].min().item(),
                                        args.model + ' b max': model.ellipse[model.hidden_dim:].max().item(),
                                    }
                                )
                            logs.append(log)
                            logs_rel[rel].append(log)

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
            metrics_rel = defaultdict(dict)
            metrics_rel_h, metrics_rel_t = defaultdict(dict), defaultdict(dict)
            for rel in logs_rel:
                for metric in logs_rel[rel][0].keys():
                    metrics_rel[rel][metric] = sum([log[metric] for log in logs_rel[rel]]) / len(logs_rel[rel])

            if args.save_ranks:
                triple_frame.ranks = (triple_frame.loc[:, 'head-batch'] + triple_frame.loc[:, 'tail-batch'])/2
                triple_frame.to_csv('triples_ranked_'+args.data_path.split('/')[1]+args.model+'.csv')

        return metrics, metrics_rel

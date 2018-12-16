# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from logging import getLogger
import scipy
import scipy.linalg
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from .utils import get_optimizer, load_embeddings, normalize_embeddings, export_embeddings
from .utils import clip_parameters
from .dico_builder import build_dictionary
from .evaluation.word_translation import DIC_EVAL_PATH, load_identical_char_dico, load_dictionary


logger = getLogger()


class Trainer(object):

    def __init__(self, src_emb, tgt_emb, mapping, discriminator, params):
        """
        Initialize trainer script.
        """
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.src_dico = params.src_dico
        self.tgt_dico = getattr(params, 'tgt_dico', None)
        self.mapping = mapping
        self.discriminator = discriminator
        self.params = params
        self.mapping_cache = {
            'WA': None,
            'WB': None,
            'S': None,
            'W': None
        }
        # optimizers
        if hasattr(params, 'map_optimizer'):
            optim_fn, optim_params = get_optimizer(params.map_optimizer)
            self.map_optimizer = optim_fn(mapping.parameters(), **optim_params)
        if hasattr(params, 'dis_optimizer'):
            optim_fn, optim_params = get_optimizer(params.dis_optimizer)
            self.dis_optimizer = optim_fn(discriminator.parameters(), **optim_params)
        else:
            assert discriminator is None

        # best validation score
        self.best_valid_metric = -1e12

        self.decrease_lr = False

    def get_dis_xy(self, volatile):
        """
        Get discriminator input batch / output target.
        """
        # select random word IDs
        def whitening(src, tgt):
            src1 = src.data.cpu().numpy()
            tgt1 = tgt.data.cpu().numpy()
            U_a, S_a, Vt_a = scipy.linalg.svd(src1, full_matrices=False)
            W_A1 = Vt_a.T.dot(np.diag(1 / S_a).dot(Vt_a))
            U_b, S_b, Vt_b = scipy.linalg.svd(tgt1, full_matrices=False)
            W_B1 = Vt_b.T.dot(np.diag(1 / S_b).dot(Vt_b))
            src1 = src1.dot(W_A1)
            tgt1 = tgt1.dot(W_B1)

            src1 = Variable(torch.from_numpy(src1), volatile=True)
            tgt1 = Variable(torch.from_numpy(tgt1), volatile=True)
            return src1, tgt1

        bs = self.params.batch_size
        mf = self.params.dis_most_frequent
        assert mf <= min(len(self.src_dico), len(self.tgt_dico))
        src_ids = torch.LongTensor(bs).random_(len(self.src_dico) if mf == 0 else mf)
        tgt_ids = torch.LongTensor(bs).random_(len(self.tgt_dico) if mf == 0 else mf)
        if self.params.cuda:
            src_ids = src_ids.cuda()
            tgt_ids = tgt_ids.cuda()

        # get word embeddings
        src_emb = torch.normal(1 * self.src_emb(Variable(src_ids, volatile=True)), 0.5)
        tgt_emb = torch.normal(1 * self.tgt_emb(Variable(tgt_ids, volatile=True)), 0.5)
        src_emb, tgt_emb = whitening(src_emb, tgt_emb)
        src_emb = self.mapping(Variable(src_emb.data, volatile=volatile))
        tgt_emb = Variable(tgt_emb.data, volatile=volatile)

        # input / target
        x = torch.cat([src_emb, tgt_emb], 0)
        y = torch.FloatTensor(2 * bs).zero_()
        y[:bs] = 1 - self.params.dis_smooth
        y[bs:] = self.params.dis_smooth
        y = Variable(y.cuda() if self.params.cuda else y)

        return x, y

    def dis_step(self, stats):
        """
        Train the discriminator.
        """
        self.discriminator.train()

        # loss
        x, y = self.get_dis_xy(volatile=True)
        preds = self.discriminator(Variable(x.data))
        loss = F.binary_cross_entropy(preds, y)
        stats['DIS_COSTS'].append(loss.data[0])

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (discriminator)")
            exit()

        # optim
        self.dis_optimizer.zero_grad()
        loss.backward()
        self.dis_optimizer.step()
        clip_parameters(self.discriminator, self.params.dis_clip_weights)

    def mapping_step(self, stats):
        """
        Fooling discriminator training step.
        """
        if self.params.dis_lambda == 0:
            return 0

        self.discriminator.eval()

        # loss
        x, y = self.get_dis_xy(volatile=False)
        preds = self.discriminator(x)
        loss = F.binary_cross_entropy(preds, 1 - y)
        loss = self.params.dis_lambda * loss

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected (fool discriminator)")
            exit()

        # optim
        self.map_optimizer.zero_grad()
        loss.backward()
        self.map_optimizer.step()
        self.orthogonalize()

        return 2 * self.params.batch_size

    def load_training_dico(self, dico_train):
        """
        Load training dictionary.
        """
        word2id1 = self.src_dico.word2id
        word2id2 = self.tgt_dico.word2id

        # identical character strings
        if dico_train == "identical_char":
            self.dico = load_identical_char_dico(word2id1, word2id2)
        # use one of the provided dictionary
        elif dico_train == "default":
            filename = '%s-%s.0-5000.txt' % (self.params.src_lang, self.params.tgt_lang)
            self.dico = load_dictionary(
                os.path.join(DIC_EVAL_PATH, filename),
                word2id1, word2id2
            )
        # dictionary provided by the user
        else:
            self.dico = load_dictionary(dico_train, word2id1, word2id2)

        # cuda
        if self.params.cuda:
            self.dico = self.dico.cuda()

    def build_dictionary(self):
        """
        Build a dictionary from aligned embeddings.
        """
        # src_emb = self.mapping(self.src_emb.weight).data
        # tgt_emb = self.tgt_emb.weight.data
        src_emb, tgt_emb = self.mapped(map='to_tgt')
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
        self.dico = build_dictionary(src_emb, tgt_emb, self.params)

    def procrustes(self):
        """
        Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        """
        def whitening(src, tgt):
            src1 = src.cpu().numpy()
            tgt1 = tgt.cpu().numpy()
            U_a, S_a, Vt_a = scipy.linalg.svd(src1, full_matrices=False)
            W_A1 = Vt_a.T.dot(np.diag(1 / S_a).dot(Vt_a))
            U_b, S_b, Vt_b = scipy.linalg.svd(tgt1, full_matrices=False)
            W_B1 = Vt_b.T.dot(np.diag(1 / S_b).dot(Vt_b))
            src1 = src1.dot(W_A1)
            tgt1 = tgt1.dot(W_B1)
            return src1, tgt1

        A = self.src_emb.weight.data[self.dico[:, 0]]
        B = self.tgt_emb.weight.data[self.dico[:, 1]]
        # M = B.transpose(0, 1).mm(A).cpu().numpy()
        src, tgt = whitening(A, B)
        M = src.T.dot(tgt)
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        # W = self.mapping.weight.data
        self.mapping_cache['WA'] = torch.from_numpy(U)
        self.mapping_cache['WB'] = torch.from_numpy(V_t.T)
        self.mapping_cache['S'] = torch.FloatTensor(S)

    def mapped(self, map):
        x = None
        y = None
        if map == 'to_tgt':
            logger.info("Map source to target space ...")
            x = self.mapping(self.src_emb.weight).data
            y = self.tgt_emb.weight.data
        elif map == 'to_shared':
            logger.info("Map source and target embeddings to the shared space ...")
            x = self.src_emb.weight.data.mm(self.mapping.weight.data.mm(self.mapping_cache['WA']))
            y = self.mapping_cache['S'] * self.tgt_emb.weight.data.mm(self.mapping_cache['WB'])
        elif map == 'row':
            x = self.src_emb.weight.data
            y = self.src_emb.weight.data
        return x, y

    def orthogonalize(self):

        """
        Orthogonalize the mapping.
        """
        if self.params.map_beta > 0:
            W = self.mapping.weight.data
            beta = self.params.map_beta
            W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))
            # U, S, V_t = scipy.linalg.svd(W, full_matrices=True)
            # W.copy_(torch.from_numpy(U.dot(V_t))).cuda()

    def update_lr(self, to_log, metric):
        """
        Update learning rate when using SGD.
        """
        if 'sgd' not in self.params.map_optimizer:
            return
        old_lr = self.map_optimizer.param_groups[0]['lr']
        new_lr = max(self.params.min_lr, old_lr * self.params.lr_decay)
        if new_lr < old_lr:
            logger.info("Decreasing learning rate: %.8f -> %.8f" % (old_lr, new_lr))
        self.map_optimizer.param_groups[0]['lr'] = new_lr

        if self.params.lr_shrink < 1 and to_log[metric] >= -1e7:
            if to_log[metric] < self.best_valid_metric:
                logger.info("Validation metric is smaller than the best: %.5f vs %.5f"
                            % (to_log[metric], self.best_valid_metric))
                # decrease the learning rate, only if this is the
                # second time the validation metric decreases
                if self.decrease_lr:
                    old_lr = self.map_optimizer.param_groups[0]['lr']
                    self.map_optimizer.param_groups[0]['lr'] *= self.params.lr_shrink
                    logger.info("Shrinking the learning rate: %.5f -> %.5f"
                                % (old_lr, self.map_optimizer.param_groups[0]['lr']))
                self.decrease_lr = True

    def save_best(self, to_log, metric, map='to_shared'):
        """
        Save the best model for the given validation metric.
        """
        # best mapping for the given validation criterion
        if to_log[metric] > self.best_valid_metric:
            # new best mapping
            self.best_valid_metric = to_log[metric]
            logger.info('* Best value for "%s": %.5f' % (metric, to_log[metric]))
            # save the mapping
            W = self.mapping.weight.data
            path = os.path.join(self.params.exp_path, 'best_mapping.pth')
            logger.info('* Saving the mapping to %s ...' % path)
            torch.save(W, path)
            if map == 'to_shared':
                weight = ['WA', 'WB', 'S']
                for i in range(len(weight)):
                    path = os.path.join(self.params.exp_path, 'best_' + weight[i] + '.pth')
                    logger.info('* Saving the ' + weight[i] + ' to %s ...' % path)
                    torch.save(self.mapping_cache[weight[i]], path)

    def reload_best(self, map='to_shared'):
        """
        Reload the best mapping.
        """
        # reload the model
        path = os.path.join(self.params.exp_path, 'best_mapping.pth')
        logger.info('* Reloading the best model from %s ...' % path)
        assert os.path.isfile(path)
        self.mapping.weight.data.copy_(torch.load(path))
        if map == 'to_shared':
            weight = ['WA', 'WB', 'S']
            for i in range(len(weight)):
                path = os.path.join(self.params.exp_path, 'best_' + weight[i] + '.pth')
                logger.info('* Reloading the best model best_' + weight[i] + ' from to %s ...' % path)
                self.mapping_cache[weight[i]].copy_(torch.load(path))


    def export(self):
        """
        Export embeddings.
        """
        params = self.params

        # load all embeddings
        logger.info("Reloading all embeddings for mapping ...")
        params.src_dico, src_emb = load_embeddings(params, source=True, full_vocab=True)
        params.tgt_dico, tgt_emb = load_embeddings(params, source=False, full_vocab=True)
        normalize_embeddings(src_emb, params.normalize_embeddings, mean=params.src_mean)
        normalize_embeddings(tgt_emb, params.normalize_embeddings, mean=params.tgt_mean)
        src_emb, tgt_emb = self.mapped()

        # write embeddings to the disk
        export_embeddings(src_emb, tgt_emb, params)

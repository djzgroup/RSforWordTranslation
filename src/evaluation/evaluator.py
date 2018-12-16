# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from copy import deepcopy
import numpy as np
from torch.autograd import Variable

from . import get_crosslingual_wordsim_scores
from . import get_word_translation_accuracy
from ..dico_builder import get_candidates, build_dictionary



logger = getLogger()


class Evaluator(object):

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.src_emb = trainer.src_emb
        self.tgt_emb = trainer.tgt_emb
        self.src_dico = trainer.src_dico
        self.tgt_dico = trainer.tgt_dico
        self.mapping = trainer.mapping
        self.discriminator = trainer.discriminator
        self.params = trainer.params
        self.trainer = trainer

    def crosslingual_wordsim(self, to_log, map):
        """
        Evaluation on cross-lingual word similarity.
        """
        # mapped word embeddings
        src_emb, tgt_emb = self.trainer.mapped(map)
        # cross-lingual wordsim evaluation
        src_tgt_ws_scores = get_crosslingual_wordsim_scores(
            self.src_dico.lang, self.src_dico.word2id, src_emb.cpu().numpy(),
            self.tgt_dico.lang, self.tgt_dico.word2id, tgt_emb.cpu().numpy(),
        )
        if src_tgt_ws_scores is None:
            return
        ws_crosslingual_scores = np.mean(list(src_tgt_ws_scores.values()))
        logger.info("Cross-lingual word similarity score average: %.5f" % ws_crosslingual_scores)
        to_log['ws_crosslingual_scores'] = ws_crosslingual_scores
        to_log.update({'src_tgt_' + k: v for k, v in src_tgt_ws_scores.items()})

    def word_translation(self, to_log, map):
        """
        Evaluation on word translation.
        """
        # mapped word embeddings
        src_emb, tgt_emb = self.trainer.mapped(map)
        results = get_word_translation_accuracy(
            self.src_dico.lang, self.src_dico.word2id, src_emb,
            self.tgt_dico.lang, self.tgt_dico.word2id, tgt_emb,
            dico_eval=self.params.dico_eval,
            exp_path=self.params.exp_path,
            alpha=self.params.alpha,
            inv_K=self.params.inv_K
        )
        to_log.update([('%s-%s' % (k, 'rs'), v) for k, v in results])


    def dist_mean_cosine(self, to_log, map):
        """
        Mean-cosine model selection criterion.
        """
        # get normalized embeddings
        src_emb, tgt_emb = self.trainer.mapped(map)
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)

        dico_build = 'S2T'
        dico_max_size = 10000
        # temp params / dictionary generation
        _params = deepcopy(self.params)
        _params.dico_build = dico_build
        _params.dico_threshold = 0
        _params.dico_max_rank = 10000
        _params.dico_min_size = 0
        _params.dico_max_size = dico_max_size
        s2t_candidates = get_candidates(src_emb, tgt_emb, _params)
        t2s_candidates = get_candidates(tgt_emb, src_emb, _params)
        dico = build_dictionary(src_emb, tgt_emb, _params, s2t_candidates, t2s_candidates)
        # mean cosine
        if dico is None:
            mean_cosine = -1e9
        else:
            mean_cosine = (src_emb[dico[:dico_max_size, 0]] * tgt_emb[dico[:dico_max_size, 1]]).sum(1).mean()
        logger.info("Mean cosine (%s method, %s build, %i max size): %.5f"
                    % ('rs', _params.dico_build, dico_max_size, mean_cosine))
        to_log['mean_cosine-%s-%s-%i' % ('rs', _params.dico_build, dico_max_size)] = mean_cosine

    def all_eval(self, to_log, map='to_shared'):
        """
        Run all evaluations.
        """
        # self.monolingual_wordsim(to_log, to_tgt)
        self.crosslingual_wordsim(to_log, map)
        self.word_translation(to_log, map)
        # self.sent_translation(to_log, to_tgt)
        self.dist_mean_cosine(to_log, map)

    def eval_dis(self, to_log):
        """
        Evaluate discriminator predictions and accuracy.
        """
        bs = 128
        src_preds = []
        tgt_preds = []

        self.discriminator.eval()

        for i in range(0, self.src_emb.num_embeddings, bs):
            emb = Variable(self.src_emb.weight[i:i + bs].data, volatile=True)
            preds = self.discriminator(self.mapping(emb))
            src_preds.extend(preds.data.cpu().tolist())

        for i in range(0, self.tgt_emb.num_embeddings, bs):
            emb = Variable(self.tgt_emb.weight[i:i + bs].data, volatile=True)
            preds = self.discriminator(emb)
            tgt_preds.extend(preds.data.cpu().tolist())

        src_pred = np.mean(src_preds)
        tgt_pred = np.mean(tgt_preds)
        logger.info("Discriminator source / target predictions: %.5f / %.5f"
                    % (src_pred, tgt_pred))

        src_accu = np.mean([x >= 0.5 for x in src_preds])
        tgt_accu = np.mean([x < 0.5 for x in tgt_preds])
        dis_accu = ((src_accu * self.src_emb.num_embeddings + tgt_accu * self.tgt_emb.num_embeddings) /
                    (self.src_emb.num_embeddings + self.tgt_emb.num_embeddings))
        logger.info("Discriminator source / target / global accuracy: %.5f / %.5f / %.5f"
                    % (src_accu, tgt_accu, dis_accu))

        to_log['dis_accu'] = dis_accu
        to_log['dis_src_pred'] = src_pred
        to_log['dis_tgt_pred'] = tgt_pred

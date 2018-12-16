# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# #
# import shutil
from logging import getLogger
import torch
from .utils import get_inv_src
import os

logger = getLogger()


def get_candidates(emb1, emb2, params):
    """
    Get best translation pairs candidates.
    """
    bs = 128
    all_scores = []
    all_targets = []

    n_src = params.dico_max_rank

    inv_src_emb = get_inv_src(emb1, emb2, params.exp_path, params.inv_K)

    for i in range(0, n_src, bs):
        query = emb1[i:min(n_src, i + bs)]
        scores1 = params.alpha * query.mm(inv_src_emb.transpose(0, 1))
        scores2 = (1 - params.alpha) * query.mm(emb2.transpose(0, 1))
        scores = scores1 + scores2
        best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)
        # update scores / potential targets
        all_scores.append(best_scores.cpu())
        all_targets.append(best_targets.cpu())

    all_scores = torch.cat(all_scores, 0)
    all_targets = torch.cat(all_targets, 0)

    all_pairs = torch.cat([
        torch.arange(0, all_targets.size(0)).long().unsqueeze(1),
        all_targets[:, 0].unsqueeze(1)
    ], 1)

    # sanity check
    assert all_scores.size() == all_pairs.size() == (n_src, 2)

    # sort pairs by score confidence
    diff = all_scores[:, 0] - all_scores[:, 1]
    reordered = diff.sort(0, descending=True)[1]
    all_scores = all_scores[reordered]
    all_pairs = all_pairs[reordered]

    # max dico words rank
    if params.dico_max_rank > 0:
        selected = all_pairs.max(1)[0] <= params.dico_max_rank
        mask = selected.unsqueeze(1).expand_as(all_scores).clone()
        all_scores = all_scores.masked_select(mask).view(-1, 2)
        all_pairs = all_pairs.masked_select(mask).view(-1, 2)

    # max dico size
    if params.dico_max_size > 0:
        all_scores = all_scores[:params.dico_max_size]
        all_pairs = all_pairs[:params.dico_max_size]

    # min dico size
    diff = all_scores[:, 0] - all_scores[:, 1]
    if params.dico_min_size > 0:
        diff[:params.dico_min_size] = 1e9

    # confidence threshold
    if params.dico_threshold > 0:
        mask = diff > params.dico_threshold
        logger.info("Selected %i / %i pairs above the confidence threshold." % (mask.sum(), diff.size(0)))
        mask = mask.unsqueeze(1).expand_as(all_pairs).clone()
        all_pairs = all_pairs.masked_select(mask).view(-1, 2)
    return all_pairs


def build_dictionary(src_emb, tgt_emb, params, s2t_candidates=None, t2s_candidates=None):
    """
    Build a training dictionary given current embeddings / mapping.
    """
    logger.info("Building the train dictionary ...")
    s2t = 'S2T' in params.dico_build
    t2s = 'T2S' in params.dico_build
    assert s2t or t2s

    if t2s:
        if t2s_candidates is None:
            t2s_candidates = get_candidates(tgt_emb, src_emb, params)
        t2s_candidates = torch.cat([t2s_candidates[:, 1:], t2s_candidates[:, :1]], 1)

    if s2t:
        if s2t_candidates is None:
            s2t_candidates = get_candidates(src_emb, tgt_emb, params)

    if params.dico_build == 'S2T':
        dico = s2t_candidates
    elif params.dico_build == 'T2S':
        dico = t2s_candidates
    else:
        s2t_candidates = set([(a, b) for a, b in s2t_candidates])
        t2s_candidates = set([(a, b) for a, b in t2s_candidates])
        if params.dico_build == 'S2T|T2S':
            final_pairs = s2t_candidates | t2s_candidates
        else:
            assert params.dico_build == 'S2T&T2S'
            final_pairs = s2t_candidates & t2s_candidates
            if len(final_pairs) == 0:
                logger.warning("Empty intersection ...")
                return None
        dico = torch.LongTensor(list([[a, b] for (a, b) in final_pairs]))

    logger.info('New train dictionary of %i pairs.' % dico.size(0))
    return dico.cuda() if params.cuda else dico
